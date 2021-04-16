const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;
const math = std.math;
const DW = std.dwarf;
const leb128 = std.leb;
const assert = std.debug.assert;
const log = std.log.scoped(.codegen);
const Module = @import("../Module.zig");
const LazySrcLoc = Module.LazySrcLoc;
const ErrorMsg = Module.ErrorMsg;
const ir = @import("../ir.zig");
const Type = @import("../type.zig").Type;
const TypedValue = @import("../TypedValue.zig");
const link = @import("../link.zig");
const Codegen = @import("../codegen.zig");
const DebugInfoOutput = Codegen.DebugInfoOutput;

const ExampleFunction = struct {
    gpa: *Allocator,
    target: *const std.Target,
    code: *std.ArrayList(u8),
    debug_output: DebugInfoOutput,
    err_msg: ?*ErrorMsg,

    /// Byte offset within the source file.
    prev_di_src: usize,
    /// Relative to the beginning of `code`.
    prev_di_pc: usize,
    /// Used to find newlines and count line deltas.
    source: []const u8,

    /// Whenever there is a runtime branch, we push a Branch onto this stack,
    /// and pop it off when the runtime branch joins. This provides an "overlay"
    /// of the table of mappings from instructions to `MCValue` from within the branch.
    /// This way we can modify the `MCValue` for an instruction in different ways
    /// within different branches. Special consideration is needed when a branch
    /// joins with its parent, to make sure all instructions have the same MCValue
    /// across each runtime branch upon joining.
    branch_stack: *std.ArrayList(Branch),

    register_manager: RegisterManager(Self, getRegisterType(), &callee_preserved_regs) = .{},

    /// Maps offset to what is stored there.
    stack: std.AutoHashMapUnmanaged(u32, StackAllocation) = .{},

    /// Offset from the stack base, representing the end of the stack frame.
    max_end_stack: u32 = 0,

    const Self = @This();

    // So we can use comptime known constants
    pub const arch: std.Target.Cpu.Arch = .x86_64;

    // Replace .x86_64 with your own arch
    pub const Branch = Codegen.Branch(.x86_64);

    // Architecture dependent
    pub fn getRegisterType() type {}
    pub const callee_preserved_regs: [0]getRegisterType() = .{};

    // At the minimum, must contain the following tags
    pub const MCValue = union(enum) {
        /// No runtime bits. `void` types, empty structs, u0, enums with 1 tag, etc.
        /// TODO Look into deleting this tag and using `dead` instead, since every use
        /// of MCValue.none should be instead looking at the type and noticing it is 0 bits.
        none,
        /// The value is undefined.
        undef,
        /// A pointer-sized integer that fits in a register.
        /// If the type is a pointer, this is the pointer address in virtual address space.
        immediate: u64,
        /// The value is in a target-specific register.
        register: getRegisterType(),
        /// The value is in memory at a hard-coded address.
        /// If the type is a pointer, it means the pointer address is at this memory location.
        memory: u64,
        /// The value is one of the stack variables.
        /// If the type is a pointer, it means the pointer address is in the stack at this offset.
        stack_offset: u32,
    };

    /// Generates code that computes `inst` into `self.code`, and returns the
    /// MCValue pointing to it
    pub fn genFuncInst(self: *Self, inst: *ir.Inst) !MCValue {
        @panic("Interface called");
    }
};

const InnerError = error{
    OutOfMemory,
    CodegenFail,
};

/// See `ExampleFunction` for what fields/decls Function must have for this to work.
pub fn genBody(comptime Function: type, self: *Function, body: ir.Body) InnerError!void {
    for (body.instructions) |inst| {
        try ensureProcessDeathCapacity(Function, self, @popCount(@TypeOf(inst.deaths), inst.deaths));

        const mcv = try self.genFuncInst(inst);
        if (!inst.isUnused()) {
            log.debug("{*} => {}", .{ inst, mcv });
            const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
            try branch.inst_table.putNoClobber(self.gpa, inst, mcv);
        }

        var i: ir.Inst.DeathsBitIndex = 0;
        while (inst.getOperand(i)) |operand| : (i += 1) {
            if (inst.operandDies(i))
                processDeath(Function, self, operand);
        }
    }
}

fn genTypedValue(comptime Function: type, self: *Function, src: LazySrcLoc, typed_value: TypedValue) InnerError!Function.MCValue {
    if (typed_value.val.isUndef())
        return Function.MCValue{ .undef = {} };
    const ptr_bits = Function.arch.ptrBitWidth();
    const ptr_bytes: u64 = @divExact(ptr_bits, 8);
    switch (typed_value.ty.zigTypeTag()) {
        .Pointer => {
            if (typed_value.val.castTag(.decl_ref)) |payload| {
                if (self.bin_file.cast(link.File.Elf)) |elf_file| {
                    const decl = payload.data;
                    const got = &elf_file.program_headers.items[elf_file.phdr_got_index.?];
                    const got_addr = got.p_vaddr + decl.link.elf.offset_table_index * ptr_bytes;
                    return Function.MCValue{ .memory = got_addr };
                } else if (self.bin_file.cast(link.File.MachO)) |macho_file| {
                    const decl = payload.data;
                    const got_addr = blk: {
                        const seg = macho_file.load_commands.items[macho_file.data_const_segment_cmd_index.?].Segment;
                        const got = seg.sections.items[macho_file.got_section_index.?];
                        break :blk got.addr + decl.link.macho.offset_table_index * ptr_bytes;
                    };
                    return Function.MCValue{ .memory = got_addr };
                } else if (self.bin_file.cast(link.File.Coff)) |coff_file| {
                    const decl = payload.data;
                    const got_addr = coff_file.offset_table_virtual_address + decl.link.coff.offset_table_index * ptr_bytes;
                    return Function.MCValue{ .memory = got_addr };
                } else {
                    return fail(Function, self, src, "TODO codegen non-ELF const Decl pointer", .{});
                }
            }
            return fail(Function, self, src, "TODO codegen more kinds of const pointers", .{});
        },
        .Int => {
            const info = typed_value.ty.intInfo(self.target.*);
            if (info.bits > ptr_bits or info.signedness == .signed) {
                return fail(Function, self, src, "TODO const int bigger than ptr and signed int", .{});
            }
            return Function.MCValue{ .immediate = typed_value.val.toUnsignedInt() };
        },
        .Bool => {
            return Function.MCValue{ .immediate = @boolToInt(typed_value.val.toBool()) };
        },
        .ComptimeInt => unreachable, // semantic analysis prevents this
        .ComptimeFloat => unreachable, // semantic analysis prevents this
        .Optional => {
            if (typed_value.ty.isPtrLikeOptional()) {
                if (typed_value.val.isNull())
                    return Function.MCValue{ .immediate = 0 };

                var buf: Type.Payload.ElemType = undefined;
                return genTypedValue(Function, self, src, .{
                    .ty = typed_value.ty.optionalChild(&buf),
                    .val = typed_value.val,
                });
            } else if (typed_value.ty.abiSize(self.target.*) == 1) {
                return Function.MCValue{ .immediate = @boolToInt(typed_value.val.isNull()) };
            }
            return fail(Function, self, src, "TODO non pointer optionals", .{});
        },
        else => return fail(Function, self, src, "TODO implement const of type '{}'", .{typed_value.ty}),
    }
}

/// Asserts there is already capacity to insert into top branch inst_table.
pub fn processDeath(comptime Function: type, self: *Function, inst: *ir.Inst) void {
    if (inst.tag == .constant) return; // Constants are immortal.
    // When editing this function, note that the logic must synchronize with `reuseOperand`.
    const prev_value = getResolvedInstValue(Function, self, inst);
    const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
    branch.inst_table.putAssumeCapacity(inst, .dead);
    switch (prev_value) {
        .register => |reg| {
            const canon_reg = Function.toCanonicalReg(reg);
            self.register_manager.freeReg(canon_reg);
        },
        else => {}, // TODO process stack allocation death
    }
}

pub fn ensureProcessDeathCapacity(comptime Function: type, self: *Function, additional_count: usize) !void {
    const table = &self.branch_stack.items[self.branch_stack.items.len - 1].inst_table;
    try table.ensureCapacity(self.gpa, table.items().len + additional_count);
}

pub fn dbgSetPrologueEnd(comptime Function: type, self: *Function) InnerError!void {
    switch (self.debug_output) {
        .dwarf => |dbg_out| {
            try dbg_out.dbg_line.append(DW.LNS_set_prologue_end);
            try dbgAdvancePCAndLine(Function, self, self.prev_di_src);
        },
        .none => {},
    }
}

pub fn dbgSetEpilogueBegin(comptime Function: type, self: *Function) InnerError!void {
    switch (self.debug_output) {
        .dwarf => |dbg_out| {
            try dbg_out.dbg_line.append(DW.LNS_set_epilogue_begin);
            try dbgAdvancePCAndLine(Function, self, self.prev_di_src);
        },
        .none => {},
    }
}

pub fn dbgAdvancePCAndLine(comptime Function: type, self: *Function, abs_byte_off: usize) InnerError!void {
    self.prev_di_src = abs_byte_off;
    self.prev_di_pc = self.code.items.len;
    switch (self.debug_output) {
        .dwarf => |dbg_out| {
            // TODO Look into improving the performance here by adding a token-index-to-line
            // lookup table, and changing ir.Inst from storing byte offset to token. Currently
            // this involves scanning over the source code for newlines
            // (but only from the previous byte offset to the new one).
            const delta_line = std.zig.lineDelta(self.source, self.prev_di_src, abs_byte_off);
            const delta_pc = self.code.items.len - self.prev_di_pc;
            // TODO Look into using the DWARF special opcodes to compress this data. It lets you emit
            // single-byte opcodes that add different numbers to both the PC and the line number
            // at the same time.
            try dbg_out.dbg_line.ensureCapacity(dbg_out.dbg_line.items.len + 11);
            dbg_out.dbg_line.appendAssumeCapacity(DW.LNS_advance_pc);
            leb128.writeULEB128(dbg_out.dbg_line.writer(), delta_pc) catch unreachable;
            if (delta_line != 0) {
                dbg_out.dbg_line.appendAssumeCapacity(DW.LNS_advance_line);
                leb128.writeILEB128(dbg_out.dbg_line.writer(), delta_line) catch unreachable;
            }
            dbg_out.dbg_line.appendAssumeCapacity(DW.LNS_copy);
        },
        .none => {},
    }
}

/// Adds a Type to the .debug_info at the current position. The bytes will be populated later,
/// after codegen for this symbol is done.
pub fn addDbgInfoTypeReloc(comptime Function: type, self: *Function, ty: Type) !void {
    switch (self.debug_output) {
        .dwarf => |dbg_out| {
            assert(ty.hasCodeGenBits());
            const index = dbg_out.dbg_info.items.len;
            try dbg_out.dbg_info.resize(index + 4); // DW.AT_type,  DW.FORM_ref4

            const gop = try dbg_out.dbg_info_type_relocs.getOrPut(self.gpa, ty);
            if (!gop.found_existing) {
                gop.entry.value = .{
                    .off = undefined,
                    .relocs = .{},
                };
            }
            try gop.entry.value.relocs.append(self.gpa, @intCast(u32, index));
        },
        .none => {},
    }
}

pub fn resolveInst(comptime Function: type, self: *Function, inst: *ir.Inst) !Function.MCValue {
    // If the type has no codegen bits, no need to store it.
    if (!inst.ty.hasCodeGenBits())
        return Function.MCValue.none;

    // Constants have static lifetimes, so they are always memoized in the outer most table.
    if (inst.castTag(.constant)) |const_inst| {
        const branch = &self.branch_stack.items[0];
        const gop = try branch.inst_table.getOrPut(self.gpa, inst);
        if (!gop.found_existing) {
            gop.entry.value = try genTypedValue(Function, self, inst.src, .{ .ty = inst.ty, .val = const_inst.val });
        }
        return gop.entry.value;
    }

    return getResolvedInstValue(Function, self, inst);
}

pub fn getResolvedInstValue(comptime Function: type, self: *Function, inst: *ir.Inst) Function.MCValue {
    // Treat each stack item as a "layer" on top of the previous one.
    var i: usize = self.branch_stack.items.len;
    while (true) {
        i -= 1;
        if (self.branch_stack.items[i].inst_table.get(inst)) |mcv| {
            assert(mcv != .dead);
            return mcv;
        }
    }
}

pub fn allocMem(comptime Function: type, self: *Function, inst: *ir.Inst, abi_size: u32, abi_align: u32) !u32 {
    if (abi_align > self.stack_align)
        self.stack_align = abi_align;
    // TODO find a free slot instead of always appending
    const offset = mem.alignForwardGeneric(u32, self.next_stack_offset, abi_align);
    self.next_stack_offset = offset + abi_size;
    if (self.next_stack_offset > self.max_end_stack)
        self.max_end_stack = self.next_stack_offset;
    try self.stack.putNoClobber(self.gpa, offset, .{
        .inst = inst,
        .size = abi_size,
    });
    return offset;
}

/// Use a pointer instruction as the basis for allocating stack memory.
pub fn allocMemPtr(comptime Function: type, self: *Function, inst: *ir.Inst) !u32 {
    const elem_ty = inst.ty.elemType();
    const abi_size = math.cast(u32, elem_ty.abiSize(self.target.*)) catch {
        return fail(Function, self, inst.src, "type '{}' too big to fit into stack frame", .{elem_ty});
    };
    // TODO swap this for inst.ty.ptrAlign
    const abi_align = elem_ty.abiAlignment(self.target.*);
    return allocMem(Function, self, inst, abi_size, abi_align);
}

pub fn allocRegOrMem(comptime Function: type, self: *Function, inst: *ir.Inst, reg_ok: bool) !Function.MCValue {
    const elem_ty = inst.ty;
    const abi_size = math.cast(u32, elem_ty.abiSize(self.target.*)) catch {
        return fail(Function, self, inst.src, "type '{}' too big to fit into stack frame", .{elem_ty});
    };
    const abi_align = elem_ty.abiAlignment(self.target.*);
    if (abi_align > self.stack_align)
        self.stack_align = abi_align;

    if (reg_ok) {
        // Make sure the type can fit in a register before we try to allocate one.
        const ptr_bits = Function.arch.ptrBitWidth();
        const ptr_bytes: u64 = @divExact(ptr_bits, 8);
        if (abi_size <= ptr_bytes) {
            try self.register_manager.registers.ensureCapacity(self.gpa, self.register_manager.registers.count() + 1);
            if (self.register_manager.tryAllocReg(inst)) |reg| {
                return Function.MCValue{ .register = Function.registerAlias(reg, abi_size) };
            }
        }
    }
    const stack_offset = try allocMem(Function, self, inst, abi_size, abi_align);
    return Function.MCValue{ .stack_offset = stack_offset };
}

pub fn fail(comptime Function: type, self: *Function, src: LazySrcLoc, comptime format: []const u8, args: anytype) InnerError {
    @setCold(true);
    assert(self.err_msg == null);
    const src_loc = if (src != .unneeded)
        src.toSrcLocWithDecl(self.mod_fn.owner_decl)
    else
        self.src_loc;
    self.err_msg = try ErrorMsg.create(self.bin_file.allocator, src_loc, format, args);
    return error.CodegenFail;
}

pub fn failSymbol(comptime Function: type, self: *Function, comptime format: []const u8, args: anytype) InnerError {
    @setCold(true);
    assert(self.err_msg == null);
    self.err_msg = try ErrorMsg.create(self.bin_file.allocator, self.src_loc, format, args);
    return error.CodegenFail;
}

/// Copies a value to a register without tracking the register. The register is not considered
/// allocated. A second call to `copyToTmpRegister` may return the same register.
/// This can have a side effect of spilling instructions to the stack to free up a register.
pub fn copyToTmpRegister(comptime Function: type, self: *Function, src: LazySrcLoc, ty: Type, mcv: Function.MCValue) !Function.getRegisterType() {
    const reg = try self.register_manager.allocRegWithoutTracking();
    try self.genSetReg(src, ty, reg, mcv);
    return reg;
}

/// Allocates a new register and copies `mcv` into it.
/// `reg_owner` is the instruction that gets associated with the register in the register table.
/// This can have a side effect of spilling instructions to the stack to free up a register.
pub fn copyToNewRegister(comptime Function: type, self: *Function, reg_owner: *ir.Inst, mcv: Function.MCValue) !Function.MCValue {
    try self.register_manager.registers.ensureCapacity(self.gpa, @intCast(u32, self.register_manager.registers.count() + 1));

    const reg = try self.register_manager.allocReg(reg_owner);
    try self.genSetReg(reg_owner.src, reg_owner.ty, reg, mcv);
    return Function.MCValue{ .register = reg };
}

/// Sets the value without any modifications to register allocation metadata or stack allocation metadata.
pub fn setRegOrMem(comptime Function: type, self: *Function, src: LazySrcLoc, ty: Type, loc: Function.MCValue, val: Function.MCValue) !void {
    switch (loc) {
        .none => return,
        .register => |reg| return self.genSetReg(src, ty, reg, val),
        .stack_offset => |off| return self.genSetStack(src, ty, off, val),
        .memory => {
            return fail(Function, self, src, "TODO implement setRegOrMem for memory", .{});
        },
        else => unreachable,
    }
}
