const std = @import("std");

pub const Interpreter = @import("spu-mk2/interpreter.zig").Interpreter;

pub const ExecutionCondition = enum(u3) {
    always = 0,
    when_zero = 1,
    not_zero = 2,
    greater_zero = 3,
    less_than_zero = 4,
    greater_or_equal_zero = 5,
    less_or_equal_zero = 6,
    overflow = 7,
};

pub const InputBehaviour = enum(u2) {
    zero = 0,
    immediate = 1,
    peek = 2,
    pop = 3,
};

pub const OutputBehaviour = enum(u2) {
    discard = 0,
    push = 1,
    jump = 2,
    jump_relative = 3,
};

pub const Command = enum(u5) {
    copy = 0,
    ipget = 1,
    get = 2,
    set = 3,
    store8 = 4,
    store16 = 5,
    load8 = 6,
    load16 = 7,
    undefined0 = 8,
    undefined1 = 9,
    frget = 10,
    frset = 11,
    bpget = 12,
    bpset = 13,
    spget = 14,
    spset = 15,
    add = 16,
    sub = 17,
    mul = 18,
    div = 19,
    mod = 20,
    @"and" = 21,
    @"or" = 22,
    xor = 23,
    not = 24,
    signext = 25,
    rol = 26,
    ror = 27,
    bswap = 28,
    asr = 29,
    lsl = 30,
    lsr = 31,
};

pub const Instruction = packed struct {
    condition: ExecutionCondition,
    input0: InputBehaviour,
    input1: InputBehaviour,
    modify_flags: bool,
    output: OutputBehaviour,
    command: Command,
    reserved: u1 = 0,

    pub fn format(instr: Instruction, comptime fmt: []const u8, options: std.fmt.FormatOptions, out: anytype) !void {
        try std.fmt.format(out, "0x{x:0<4} ", .{@bitCast(u16, instr)});
        try out.writeAll(switch (instr.condition) {
            .always => "    ",
            .when_zero => "== 0",
            .not_zero => "!= 0",
            .greater_zero => " > 0",
            .less_than_zero => " < 0",
            .greater_or_equal_zero => ">= 0",
            .less_or_equal_zero => "<= 0",
            .overflow => "ovfl",
        });
        try out.writeAll(" ");
        try out.writeAll(switch (instr.input0) {
            .zero => "zero",
            .immediate => "imm ",
            .peek => "peek",
            .pop => "pop ",
        });
        try out.writeAll(" ");
        try out.writeAll(switch (instr.input1) {
            .zero => "zero",
            .immediate => "imm ",
            .peek => "peek",
            .pop => "pop ",
        });
        try out.writeAll(" ");
        try out.writeAll(switch (instr.command) {
            .copy => "copy     ",
            .ipget => "ipget    ",
            .get => "get      ",
            .set => "set      ",
            .store8 => "store8   ",
            .store16 => "store16  ",
            .load8 => "load8    ",
            .load16 => "load16   ",
            .undefined0 => "undefined",
            .undefined1 => "undefined",
            .frget => "frget    ",
            .frset => "frset    ",
            .bpget => "bpget    ",
            .bpset => "bpset    ",
            .spget => "spget    ",
            .spset => "spset    ",
            .add => "add      ",
            .sub => "sub      ",
            .mul => "mul      ",
            .div => "div      ",
            .mod => "mod      ",
            .@"and" => "and      ",
            .@"or" => "or       ",
            .xor => "xor      ",
            .not => "not      ",
            .signext => "signext  ",
            .rol => "rol      ",
            .ror => "ror      ",
            .bswap => "bswap    ",
            .asr => "asr      ",
            .lsl => "lsl      ",
            .lsr => "lsr      ",
        });
        try out.writeAll(" ");
        try out.writeAll(switch (instr.output) {
            .discard => "discard",
            .push => "push   ",
            .jump => "jmp    ",
            .jump_relative => "rjmp   ",
        });
        try out.writeAll(" ");
        try out.writeAll(if (instr.modify_flags)
            "+ flags"
        else
            "       ");
    }
};

pub const FlagRegister = packed struct {
    zero: bool,
    negative: bool,
    carry: bool,
    carry_enabled: bool,
    interrupt0_enabled: bool,
    interrupt1_enabled: bool,
    interrupt2_enabled: bool,
    interrupt3_enabled: bool,
    reserved: u8 = 0,
};

pub const Register = enum {
    dummy,

    pub fn allocIndex(self: Register) ?u4 {
        return null;
    }
};

pub const callee_preserved_regs = [_]Register{};

const mem = std.mem;
const math = std.math;
const assert = std.debug.assert;
const ir = @import("../ir.zig");
const Type = @import("../type.zig").Type;
const Value = @import("../value.zig").Value;
const TypedValue = @import("../TypedValue.zig");
const link = @import("../link.zig");
const Module = @import("../Module.zig");
const Compilation = @import("../Compilation.zig");
const ErrorMsg = Module.ErrorMsg;
const Target = std.Target;
const Allocator = mem.Allocator;
const trace = @import("../tracy.zig").trace;
const DW = std.dwarf;
const leb128 = std.leb;
const log = std.log.scoped(.codegen);
const build_options = @import("build_options");
const LazySrcLoc = Module.LazySrcLoc;
const RegisterManager = @import("../register_manager.zig").RegisterManager;

const Codegen = @import("../codegen.zig");
const BlockData = Codegen.BlockData;
const AnyMCValue = Codegen.AnyMCValue;
const Reloc = Codegen.Reloc;
const Result = Codegen.Result;
const GenerateSymbolError = Codegen.GenerateSymbolError;
const DebugInfoOutput = Codegen.DebugInfoOutput;

const arch = std.Target.Cpu.Arch.spu_2;

const InnerError = error{
    OutOfMemory,
    CodegenFail,
};

const writeInt = switch (arch.endian()) {
    .Little => mem.writeIntLittle,
    .Big => mem.writeIntBig,
};

pub const Function = struct {
    gpa: *Allocator,
    bin_file: *link.File,
    target: *const std.Target,
    mod_fn: *const Module.Fn,
    code: *std.ArrayList(u8),
    debug_output: DebugInfoOutput,
    err_msg: ?*ErrorMsg,
    args: []MCValue,
    ret_mcv: MCValue,
    fn_type: Type,
    arg_index: usize,
    src_loc: Module.SrcLoc,
    stack_align: u32,

    /// Byte offset within the source file.
    prev_di_src: usize,
    /// Relative to the beginning of `code`.
    prev_di_pc: usize,
    /// Used to find newlines and count line deltas.
    source: []const u8,
    /// Byte offset within the source file of the ending curly.
    rbrace_src: usize,

    /// The value is an offset into the `Function` `code` from the beginning.
    /// To perform the reloc, write 32-bit signed little-endian integer
    /// which is a relative jump, based on the address following the reloc.
    exitlude_jump_relocs: std.ArrayListUnmanaged(usize) = .{},

    /// Whenever there is a runtime branch, we push a Branch onto this stack,
    /// and pop it off when the runtime branch joins. This provides an "overlay"
    /// of the table of mappings from instructions to `MCValue` from within the branch.
    /// This way we can modify the `MCValue` for an instruction in different ways
    /// within different branches. Special consideration is needed when a branch
    /// joins with its parent, to make sure all instructions have the same MCValue
    /// across each runtime branch upon joining.
    branch_stack: *std.ArrayList(Branch),

    register_manager: RegisterManager(Self, Register, &callee_preserved_regs) = .{},
    /// Maps offset to what is stored there.
    stack: std.AutoHashMapUnmanaged(u32, StackAllocation) = .{},

    /// Offset from the stack base, representing the end of the stack frame.
    max_end_stack: u32 = 0,
    /// Represents the current end stack offset. If there is no existing slot
    /// to place a new stack allocation, it goes here, and then bumps `max_end_stack`.
    next_stack_offset: u32 = 0,

    const MCValue = union(enum) {
        /// No runtime bits. `void` types, empty structs, u0, enums with 1 tag, etc.
        /// TODO Look into deleting this tag and using `dead` instead, since every use
        /// of MCValue.none should be instead looking at the type and noticing it is 0 bits.
        none,
        /// Control flow will not allow this value to be observed.
        unreach,
        /// No more references to this value remain.
        dead,
        /// The value is undefined.
        undef,
        /// A pointer-sized integer that fits in a register.
        /// If the type is a pointer, this is the pointer address in virtual address space.
        immediate: u64,
        /// The constant was emitted into the code, at this offset.
        /// If the type is a pointer, it means the pointer address is embedded in the code.
        embedded_in_code: usize,
        /// The value is a pointer to a constant which was emitted into the code, at this offset.
        ptr_embedded_in_code: usize,
        /// The value is in a target-specific register.
        register: Register,
        /// The value is in memory at a hard-coded address.
        /// If the type is a pointer, it means the pointer address is at this memory location.
        memory: u64,
        /// The value is one of the stack variables.
        /// If the type is a pointer, it means the pointer address is in the stack at this offset.
        stack_offset: u32,
        /// The value is a pointer to one of the stack variables (payload is stack offset).
        ptr_stack_offset: u32,
        /// The value is in the compare flags assuming an unsigned operation,
        /// with this operator applied on top of it.
        compare_flags_unsigned: math.CompareOperator,
        /// The value is in the compare flags assuming a signed operation,
        /// with this operator applied on top of it.
        compare_flags_signed: math.CompareOperator,

        fn isMemory(mcv: MCValue) bool {
            return switch (mcv) {
                .embedded_in_code, .memory, .stack_offset => true,
                else => false,
            };
        }

        fn isImmediate(mcv: MCValue) bool {
            return switch (mcv) {
                .immediate => true,
                else => false,
            };
        }

        fn isMutable(mcv: MCValue) bool {
            return switch (mcv) {
                .none => unreachable,
                .unreach => unreachable,
                .dead => unreachable,

                .immediate,
                .embedded_in_code,
                .memory,
                .compare_flags_unsigned,
                .compare_flags_signed,
                .ptr_stack_offset,
                .ptr_embedded_in_code,
                .undef,
                => false,

                .register,
                .stack_offset,
                => true,
            };
        }
    };

    const Branch = struct {
        inst_table: std.AutoArrayHashMapUnmanaged(*ir.Inst, MCValue) = .{},

        fn deinit(self: *Branch, gpa: *Allocator) void {
            self.inst_table.deinit(gpa);
            self.* = undefined;
        }
    };

    const StackAllocation = struct {
        inst: *ir.Inst,
        /// TODO do we need size? should be determined by inst.ty.abiSize()
        size: u32,
    };

    const Self = @This();

    pub fn generateSymbol(
        bin_file: *link.File,
        src_loc: Module.SrcLoc,
        typed_value: TypedValue,
        code: *std.ArrayList(u8),
        debug_output: DebugInfoOutput,
    ) GenerateSymbolError!Result {
        if (build_options.skip_non_native and std.Target.current.cpu.arch != arch) {
            @panic("Attempted to compile for architecture that was disabled by build configuration");
        }

        const module_fn = typed_value.val.castTag(.function).?.data;

        const fn_type = module_fn.owner_decl.typed_value.most_recent.typed_value.ty;

        var branch_stack = std.ArrayList(Branch).init(bin_file.allocator);
        defer {
            assert(branch_stack.items.len == 1);
            branch_stack.items[0].deinit(bin_file.allocator);
            branch_stack.deinit();
        }
        try branch_stack.append(.{});

        const src_data: struct { lbrace_src: usize, rbrace_src: usize, source: []const u8 } = blk: {
            const container_scope = module_fn.owner_decl.container;
            const tree = container_scope.file_scope.tree;
            const node_tags = tree.nodes.items(.tag);
            const node_datas = tree.nodes.items(.data);
            const token_starts = tree.tokens.items(.start);

            const fn_decl = module_fn.owner_decl.src_node;
            assert(node_tags[fn_decl] == .fn_decl);
            const block = node_datas[fn_decl].rhs;
            const lbrace_src = token_starts[tree.firstToken(block)];
            const rbrace_src = token_starts[tree.lastToken(block)];
            break :blk .{
                .lbrace_src = lbrace_src,
                .rbrace_src = rbrace_src,
                .source = tree.source,
            };
        };

        var function = Self{
            .gpa = bin_file.allocator,
            .target = &bin_file.options.target,
            .bin_file = bin_file,
            .mod_fn = module_fn,
            .code = code,
            .debug_output = debug_output,
            .err_msg = null,
            .args = undefined, // populated after `resolveCallingConventionValues`
            .ret_mcv = undefined, // populated after `resolveCallingConventionValues`
            .fn_type = fn_type,
            .arg_index = 0,
            .branch_stack = &branch_stack,
            .src_loc = src_loc,
            .stack_align = undefined,
            .prev_di_pc = 0,
            .prev_di_src = src_data.lbrace_src,
            .rbrace_src = src_data.rbrace_src,
            .source = src_data.source,
        };
        defer function.register_manager.deinit(bin_file.allocator);
        defer function.stack.deinit(bin_file.allocator);
        defer function.exitlude_jump_relocs.deinit(bin_file.allocator);

        var call_info = function.resolveCallingConventionValues(src_loc.lazy, fn_type) catch |err| switch (err) {
            error.CodegenFail => return Result{ .fail = function.err_msg.? },
            else => |e| return e,
        };
        defer call_info.deinit(&function);

        function.args = call_info.args;
        function.ret_mcv = call_info.return_value;
        function.stack_align = call_info.stack_align;
        function.max_end_stack = call_info.stack_byte_count;

        function.gen() catch |err| switch (err) {
            error.CodegenFail => return Result{ .fail = function.err_msg.? },
            else => |e| return e,
        };

        if (function.err_msg) |em| {
            return Result{ .fail = em };
        } else {
            return Result{ .appended = {} };
        }
    }

    fn gen(self: *Self) !void {
        try self.dbgSetPrologueEnd();
        try self.genBody(self.mod_fn.body);
        try self.dbgSetEpilogueBegin();

        // Drop them off at the rbrace.
        try self.dbgAdvancePCAndLine(self.rbrace_src);
    }

    fn genBody(self: *Self, body: ir.Body) InnerError!void {
        for (body.instructions) |inst| {
            try self.ensureProcessDeathCapacity(@popCount(@TypeOf(inst.deaths), inst.deaths));

            const mcv = try self.genFuncInst(inst);
            if (!inst.isUnused()) {
                log.debug("{*} => {}", .{ inst, mcv });
                const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
                try branch.inst_table.putNoClobber(self.gpa, inst, mcv);
            }

            var i: ir.Inst.DeathsBitIndex = 0;
            while (inst.getOperand(i)) |operand| : (i += 1) {
                if (inst.operandDies(i))
                    self.processDeath(operand);
            }
        }
    }

    fn dbgSetPrologueEnd(self: *Self) InnerError!void {
        switch (self.debug_output) {
            .dwarf => |dbg_out| {
                try dbg_out.dbg_line.append(DW.LNS_set_prologue_end);
                try self.dbgAdvancePCAndLine(self.prev_di_src);
            },
            .none => {},
        }
    }

    fn dbgSetEpilogueBegin(self: *Self) InnerError!void {
        switch (self.debug_output) {
            .dwarf => |dbg_out| {
                try dbg_out.dbg_line.append(DW.LNS_set_epilogue_begin);
                try self.dbgAdvancePCAndLine(self.prev_di_src);
            },
            .none => {},
        }
    }

    fn dbgAdvancePCAndLine(self: *Self, abs_byte_off: usize) InnerError!void {
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

    /// Asserts there is already capacity to insert into top branch inst_table.
    fn processDeath(self: *Self, inst: *ir.Inst) void {
        if (inst.tag == .constant) return; // Constants are immortal.
        // When editing this function, note that the logic must synchronize with `reuseOperand`.
        const prev_value = self.getResolvedInstValue(inst);
        const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
        branch.inst_table.putAssumeCapacity(inst, .dead);
        switch (prev_value) {
            .register => |reg| {
                const canon_reg = toCanonicalReg(reg);
                self.register_manager.freeReg(canon_reg);
            },
            else => {}, // TODO process stack allocation death
        }
    }

    fn ensureProcessDeathCapacity(self: *Self, additional_count: usize) !void {
        const table = &self.branch_stack.items[self.branch_stack.items.len - 1].inst_table;
        try table.ensureCapacity(self.gpa, table.items().len + additional_count);
    }

    /// Adds a Type to the .debug_info at the current position. The bytes will be populated later,
    /// after codegen for this symbol is done.
    fn addDbgInfoTypeReloc(self: *Self, ty: Type) !void {
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

    fn genFuncInst(self: *Self, inst: *ir.Inst) !MCValue {
        switch (inst.tag) {
            .add => return self.genAdd(inst.castTag(.add).?),
            .addwrap => return self.genAddWrap(inst.castTag(.addwrap).?),
            .alloc => return self.genAlloc(inst.castTag(.alloc).?),
            .arg => return self.genArg(inst.castTag(.arg).?),
            .assembly => return self.genAsm(inst.castTag(.assembly).?),
            .bitcast => return self.genBitCast(inst.castTag(.bitcast).?),
            .bit_and => return self.genBitAnd(inst.castTag(.bit_and).?),
            .bit_or => return self.genBitOr(inst.castTag(.bit_or).?),
            .block => return self.genBlock(inst.castTag(.block).?),
            .br => return self.genBr(inst.castTag(.br).?),
            .br_block_flat => return self.genBrBlockFlat(inst.castTag(.br_block_flat).?),
            .breakpoint => return self.genBreakpoint(inst.src),
            .br_void => return self.genBrVoid(inst.castTag(.br_void).?),
            .bool_and => return self.genBoolOp(inst.castTag(.bool_and).?),
            .bool_or => return self.genBoolOp(inst.castTag(.bool_or).?),
            .call => return self.genCall(inst.castTag(.call).?),
            .cmp_lt => return self.genCmp(inst.castTag(.cmp_lt).?, .lt),
            .cmp_lte => return self.genCmp(inst.castTag(.cmp_lte).?, .lte),
            .cmp_eq => return self.genCmp(inst.castTag(.cmp_eq).?, .eq),
            .cmp_gte => return self.genCmp(inst.castTag(.cmp_gte).?, .gte),
            .cmp_gt => return self.genCmp(inst.castTag(.cmp_gt).?, .gt),
            .cmp_neq => return self.genCmp(inst.castTag(.cmp_neq).?, .neq),
            .condbr => return self.genCondBr(inst.castTag(.condbr).?),
            .constant => unreachable, // excluded from function bodies
            .dbg_stmt => return self.genDbgStmt(inst.castTag(.dbg_stmt).?),
            .floatcast => return self.genFloatCast(inst.castTag(.floatcast).?),
            .intcast => return self.genIntCast(inst.castTag(.intcast).?),
            .is_non_null => return self.genIsNonNull(inst.castTag(.is_non_null).?),
            .is_non_null_ptr => return self.genIsNonNullPtr(inst.castTag(.is_non_null_ptr).?),
            .is_null => return self.genIsNull(inst.castTag(.is_null).?),
            .is_null_ptr => return self.genIsNullPtr(inst.castTag(.is_null_ptr).?),
            .is_err => return self.genIsErr(inst.castTag(.is_err).?),
            .is_err_ptr => return self.genIsErrPtr(inst.castTag(.is_err_ptr).?),
            .error_to_int => return self.genErrorToInt(inst.castTag(.error_to_int).?),
            .int_to_error => return self.genIntToError(inst.castTag(.int_to_error).?),
            .load => return self.genLoad(inst.castTag(.load).?),
            .loop => return self.genLoop(inst.castTag(.loop).?),
            .not => return self.genNot(inst.castTag(.not).?),
            .mul => return self.genMul(inst.castTag(.mul).?),
            .mulwrap => return self.genMulWrap(inst.castTag(.mulwrap).?),
            .div => return self.genDiv(inst.castTag(.div).?),
            .ptrtoint => return self.genPtrToInt(inst.castTag(.ptrtoint).?),
            .ref => return self.genRef(inst.castTag(.ref).?),
            .ret => return self.genRet(inst.castTag(.ret).?),
            .retvoid => return self.genRetVoid(inst.castTag(.retvoid).?),
            .store => return self.genStore(inst.castTag(.store).?),
            .struct_field_ptr => return self.genStructFieldPtr(inst.castTag(.struct_field_ptr).?),
            .sub => return self.genSub(inst.castTag(.sub).?),
            .subwrap => return self.genSubWrap(inst.castTag(.subwrap).?),
            .switchbr => return self.genSwitch(inst.castTag(.switchbr).?),
            .unreach => return MCValue{ .unreach = {} },
            .optional_payload => return self.genOptionalPayload(inst.castTag(.optional_payload).?),
            .optional_payload_ptr => return self.genOptionalPayloadPtr(inst.castTag(.optional_payload_ptr).?),
            .unwrap_errunion_err => return self.genUnwrapErrErr(inst.castTag(.unwrap_errunion_err).?),
            .unwrap_errunion_payload => return self.genUnwrapErrPayload(inst.castTag(.unwrap_errunion_payload).?),
            .unwrap_errunion_err_ptr => return self.genUnwrapErrErrPtr(inst.castTag(.unwrap_errunion_err_ptr).?),
            .unwrap_errunion_payload_ptr => return self.genUnwrapErrPayloadPtr(inst.castTag(.unwrap_errunion_payload_ptr).?),
            .wrap_optional => return self.genWrapOptional(inst.castTag(.wrap_optional).?),
            .wrap_errunion_payload => return self.genWrapErrUnionPayload(inst.castTag(.wrap_errunion_payload).?),
            .wrap_errunion_err => return self.genWrapErrUnionErr(inst.castTag(.wrap_errunion_err).?),
            .varptr => return self.genVarPtr(inst.castTag(.varptr).?),
            .xor => return self.genXor(inst.castTag(.xor).?),
        }
    }

    fn allocMem(self: *Self, inst: *ir.Inst, abi_size: u32, abi_align: u32) !u32 {
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
    fn allocMemPtr(self: *Self, inst: *ir.Inst) !u32 {
        const elem_ty = inst.ty.elemType();
        const abi_size = math.cast(u32, elem_ty.abiSize(self.target.*)) catch {
            return self.fail(inst.src, "type '{}' too big to fit into stack frame", .{elem_ty});
        };
        // TODO swap this for inst.ty.ptrAlign
        const abi_align = elem_ty.abiAlignment(self.target.*);
        return self.allocMem(inst, abi_size, abi_align);
    }

    fn allocRegOrMem(self: *Self, inst: *ir.Inst, reg_ok: bool) !MCValue {
        const elem_ty = inst.ty;
        const abi_size = math.cast(u32, elem_ty.abiSize(self.target.*)) catch {
            return self.fail(inst.src, "type '{}' too big to fit into stack frame", .{elem_ty});
        };
        const abi_align = elem_ty.abiAlignment(self.target.*);
        if (abi_align > self.stack_align)
            self.stack_align = abi_align;

        if (reg_ok) {
            // Make sure the type can fit in a register before we try to allocate one.
            const ptr_bits = arch.ptrBitWidth();
            const ptr_bytes: u64 = @divExact(ptr_bits, 8);
            if (abi_size <= ptr_bytes) {
                try self.register_manager.registers.ensureCapacity(self.gpa, self.register_manager.registers.count() + 1);
                if (self.register_manager.tryAllocReg(inst)) |reg| {
                    return MCValue{ .register = registerAlias(reg, abi_size) };
                }
            }
        }
        const stack_offset = try self.allocMem(inst, abi_size, abi_align);
        return MCValue{ .stack_offset = stack_offset };
    }

    pub fn spillInstruction(self: *Self, src: LazySrcLoc, reg: Register, inst: *ir.Inst) !void {
        const stack_mcv = try self.allocRegOrMem(inst, false);
        const reg_mcv = self.getResolvedInstValue(inst);
        assert(reg == toCanonicalReg(reg_mcv.register));
        const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
        try branch.inst_table.put(self.gpa, inst, stack_mcv);
        try self.genSetStack(src, inst.ty, stack_mcv.stack_offset, reg_mcv);
    }

    /// Copies a value to a register without tracking the register. The register is not considered
    /// allocated. A second call to `copyToTmpRegister` may return the same register.
    /// This can have a side effect of spilling instructions to the stack to free up a register.
    fn copyToTmpRegister(self: *Self, src: LazySrcLoc, ty: Type, mcv: MCValue) !Register {
        const reg = try self.register_manager.allocRegWithoutTracking();
        try self.genSetReg(src, ty, reg, mcv);
        return reg;
    }

    /// Allocates a new register and copies `mcv` into it.
    /// `reg_owner` is the instruction that gets associated with the register in the register table.
    /// This can have a side effect of spilling instructions to the stack to free up a register.
    fn copyToNewRegister(self: *Self, reg_owner: *ir.Inst, mcv: MCValue) !MCValue {
        try self.register_manager.registers.ensureCapacity(self.gpa, @intCast(u32, self.register_manager.registers.count() + 1));

        const reg = try self.register_manager.allocReg(reg_owner);
        try self.genSetReg(reg_owner.src, reg_owner.ty, reg, mcv);
        return MCValue{ .register = reg };
    }

    fn genAlloc(self: *Self, inst: *ir.Inst.NoOp) !MCValue {
        const stack_offset = try self.allocMemPtr(&inst.base);
        return MCValue{ .ptr_stack_offset = stack_offset };
    }

    fn genFloatCast(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement floatCast for {}", .{self.target.cpu.arch});
    }

    fn genIntCast(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        const operand = try self.resolveInst(inst.operand);
        const info_a = inst.operand.ty.intInfo(self.target.*);
        const info_b = inst.base.ty.intInfo(self.target.*);
        if (info_a.signedness != info_b.signedness)
            return self.fail(inst.base.src, "TODO gen intcast sign safety in semantic analysis", .{});

        if (info_a.bits == info_b.bits)
            return operand;

        return self.fail(inst.base.src, "TODO implement intCast for {}", .{self.target.cpu.arch});
    }

    fn genNot(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        const operand = try self.resolveInst(inst.operand);
        switch (operand) {
            .dead => unreachable,
            .unreach => unreachable,
            .compare_flags_unsigned => |op| return MCValue{
                .compare_flags_unsigned = switch (op) {
                    .gte => .lt,
                    .gt => .lte,
                    .neq => .eq,
                    .lt => .gte,
                    .lte => .gt,
                    .eq => .neq,
                },
            },
            .compare_flags_signed => |op| return MCValue{
                .compare_flags_signed = switch (op) {
                    .gte => .lt,
                    .gt => .lte,
                    .neq => .eq,
                    .lt => .gte,
                    .lte => .gt,
                    .eq => .neq,
                },
            },
            else => {},
        }

        return self.fail(inst.base.src, "TODO implement NOT for {}", .{self.target.cpu.arch});
    }

    fn genAdd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement add for {}", .{self.target.cpu.arch});
    }

    fn genAddWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement addwrap for {}", .{self.target.cpu.arch});
    }

    fn genMul(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement mul for {}", .{self.target.cpu.arch});
    }

    fn genMulWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement mulwrap for {}", .{self.target.cpu.arch});
    }

    fn genDiv(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement div for {}", .{self.target.cpu.arch});
    }

    fn genBitAnd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement bitwise and for {}", .{self.target.cpu.arch});
    }

    fn genBitOr(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement bitwise or for {}", .{self.target.cpu.arch});
    }

    fn genXor(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement xor for {}", .{self.target.cpu.arch});
    }

    fn genOptionalPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement .optional_payload for {}", .{self.target.cpu.arch});
    }

    fn genOptionalPayloadPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement .optional_payload_ptr for {}", .{self.target.cpu.arch});
    }

    fn genUnwrapErrErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union error for {}", .{self.target.cpu.arch});
    }

    fn genUnwrapErrPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union payload for {}", .{self.target.cpu.arch});
    }
    // *(E!T) -> E
    fn genUnwrapErrErrPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union error ptr for {}", .{self.target.cpu.arch});
    }
    // *(E!T) -> *T
    fn genUnwrapErrPayloadPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union payload ptr for {}", .{self.target.cpu.arch});
    }
    fn genWrapOptional(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const optional_ty = inst.base.ty;

        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        // Optional type is just a boolean true
        if (optional_ty.abiSize(self.target.*) == 1)
            return MCValue{ .immediate = 1 };

        return self.fail(inst.base.src, "TODO implement wrap optional for {}", .{self.target.cpu.arch});
    }

    /// T to E!T
    fn genWrapErrUnionPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return self.fail(inst.base.src, "TODO implement wrap errunion payload for {}", .{self.target.cpu.arch});
    }

    /// E to E!T
    fn genWrapErrUnionErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return self.fail(inst.base.src, "TODO implement wrap errunion error for {}", .{self.target.cpu.arch});
    }
    fn genVarPtr(self: *Self, inst: *ir.Inst.VarPtr) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return self.fail(inst.base.src, "TODO implement varptr for {}", .{self.target.cpu.arch});
    }

    fn reuseOperand(self: *Self, inst: *ir.Inst, op_index: ir.Inst.DeathsBitIndex, mcv: MCValue) bool {
        if (!inst.operandDies(op_index))
            return false;

        switch (mcv) {
            .register => |reg| {
                // If it's in the registers table, need to associate the register with the
                // new instruction.
                if (self.register_manager.registers.getEntry(toCanonicalReg(reg))) |entry| {
                    entry.value = inst;
                }
                log.debug("reusing {} => {*}", .{ reg, inst });
            },
            .stack_offset => |off| {
                log.debug("reusing stack offset {} => {*}", .{ off, inst });
                return true;
            },
            else => return false,
        }

        // Prevent the operand deaths processing code from deallocating it.
        inst.clearOperandDeath(op_index);

        // That makes us responsible for doing the rest of the stuff that processDeath would have done.
        const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
        branch.inst_table.putAssumeCapacity(inst.getOperand(op_index).?, .dead);

        return true;
    }

    fn genLoad(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const elem_ty = inst.base.ty;
        if (!elem_ty.hasCodeGenBits())
            return MCValue.none;
        const ptr = try self.resolveInst(inst.operand);
        const is_volatile = inst.operand.ty.isVolatilePtr();
        if (inst.base.isUnused() and !is_volatile)
            return MCValue.dead;
        const dst_mcv: MCValue = blk: {
            if (self.reuseOperand(&inst.base, 0, ptr)) {
                // The MCValue that holds the pointer can be re-used as the value.
                break :blk ptr;
            } else {
                break :blk try self.allocRegOrMem(&inst.base, true);
            }
        };
        switch (ptr) {
            .none => unreachable,
            .undef => unreachable,
            .unreach => unreachable,
            .dead => unreachable,
            .compare_flags_unsigned => unreachable,
            .compare_flags_signed => unreachable,
            .immediate => |imm| try self.setRegOrMem(inst.base.src, elem_ty, dst_mcv, .{ .memory = imm }),
            .ptr_stack_offset => |off| try self.setRegOrMem(inst.base.src, elem_ty, dst_mcv, .{ .stack_offset = off }),
            .ptr_embedded_in_code => |off| {
                try self.setRegOrMem(inst.base.src, elem_ty, dst_mcv, .{ .embedded_in_code = off });
            },
            .embedded_in_code => {
                return self.fail(inst.base.src, "TODO implement loading from MCValue.embedded_in_code", .{});
            },
            .register => {
                return self.fail(inst.base.src, "TODO implement loading from MCValue.register", .{});
            },
            .memory => {
                return self.fail(inst.base.src, "TODO implement loading from MCValue.memory", .{});
            },
            .stack_offset => {
                return self.fail(inst.base.src, "TODO implement loading from MCValue.stack_offset", .{});
            },
        }
        return dst_mcv;
    }

    fn genStore(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        const ptr = try self.resolveInst(inst.lhs);
        const value = try self.resolveInst(inst.rhs);
        const elem_ty = inst.rhs.ty;
        switch (ptr) {
            .none => unreachable,
            .undef => unreachable,
            .unreach => unreachable,
            .dead => unreachable,
            .compare_flags_unsigned => unreachable,
            .compare_flags_signed => unreachable,
            .immediate => |imm| {
                try self.setRegOrMem(inst.base.src, elem_ty, .{ .memory = imm }, value);
            },
            .ptr_stack_offset => |off| {
                try self.genSetStack(inst.base.src, elem_ty, off, value);
            },
            .ptr_embedded_in_code => |off| {
                try self.setRegOrMem(inst.base.src, elem_ty, .{ .embedded_in_code = off }, value);
            },
            .embedded_in_code => {
                return self.fail(inst.base.src, "TODO implement storing to MCValue.embedded_in_code", .{});
            },
            .register => {
                return self.fail(inst.base.src, "TODO implement storing to MCValue.register", .{});
            },
            .memory => {
                return self.fail(inst.base.src, "TODO implement storing to MCValue.memory", .{});
            },
            .stack_offset => {
                return self.fail(inst.base.src, "TODO implement storing to MCValue.stack_offset", .{});
            },
        }
        return .none;
    }

    fn genStructFieldPtr(self: *Self, inst: *ir.Inst.StructFieldPtr) !MCValue {
        return self.fail(inst.base.src, "TODO implement codegen struct_field_ptr", .{});
    }

    fn genSub(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement sub for {}", .{self.target.cpu.arch});
    }

    fn genSubWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement subwrap for {}", .{self.target.cpu.arch});
    }

    fn genArgDbgInfo(self: *Self, inst: *ir.Inst.Arg, mcv: MCValue) !void {
        const name_with_null = inst.name[0 .. mem.lenZ(inst.name) + 1];

        switch (mcv) {
            .register => |reg| {
                switch (self.debug_output) {
                    .dwarf => |dbg_out| {
                        try dbg_out.dbg_info.ensureCapacity(dbg_out.dbg_info.items.len + 3);
                        dbg_out.dbg_info.appendAssumeCapacity(link.File.Elf.abbrev_parameter);
                        dbg_out.dbg_info.appendSliceAssumeCapacity(&[2]u8{ // DW.AT_location, DW.FORM_exprloc
                            1, // ULEB128 dwarf expression length
                            reg.dwarfLocOp(),
                        });
                        try dbg_out.dbg_info.ensureCapacity(dbg_out.dbg_info.items.len + 5 + name_with_null.len);
                        try self.addDbgInfoTypeReloc(inst.base.ty); // DW.AT_type,  DW.FORM_ref4
                        dbg_out.dbg_info.appendSliceAssumeCapacity(name_with_null); // DW.AT_name, DW.FORM_string
                    },
                    .none => {},
                }
            },
            .stack_offset => {},
            else => {},
        }
    }

    fn genArg(self: *Self, inst: *ir.Inst.Arg) !MCValue {
        const arg_index = self.arg_index;
        self.arg_index += 1;

        if (callee_preserved_regs.len == 0) {
            return self.fail(inst.base.src, "TODO implement Register enum for {}", .{self.target.cpu.arch});
        }

        const result = self.args[arg_index];
        // TODO support stack-only arguments on all target architectures
        const mcv = result;
        try self.genArgDbgInfo(inst, mcv);

        if (inst.base.isUnused())
            return MCValue.dead;

        switch (mcv) {
            .register => |reg| {
                try self.register_manager.registers.ensureCapacity(self.gpa, self.register_manager.registers.count() + 1);
                self.register_manager.getRegAssumeFree(toCanonicalReg(reg), &inst.base);
            },
            else => {},
        }

        return mcv;
    }

    fn genBreakpoint(self: *Self, src: LazySrcLoc) !MCValue {
        try self.code.resize(self.code.items.len + 2);
        var instr = Instruction{ .condition = .always, .input0 = .zero, .input1 = .zero, .modify_flags = false, .output = .discard, .command = .undefined1 };
        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 2 ..][0..2], @bitCast(u16, instr));
        return .none;
    }

    fn genCall(self: *Self, inst: *ir.Inst.Call) !MCValue {
        var info = try self.resolveCallingConventionValues(inst.base.src, inst.func.ty);
        defer info.deinit(self);

        // Due to incremental compilation, how function calls are generated depends
        // on linking.
        if (self.bin_file.tag == link.File.Elf.base_tag or self.bin_file.tag == link.File.Coff.base_tag) {
            if (inst.func.value()) |func_value| {
                if (info.args.len != 0) {
                    return self.fail(inst.base.src, "TODO implement call with more than 0 parameters", .{});
                }
                if (func_value.castTag(.function)) |func_payload| {
                    const func = func_payload.data;
                    const got_addr = if (self.bin_file.cast(link.File.Elf)) |elf_file| blk: {
                        const got = &elf_file.program_headers.items[elf_file.phdr_got_index.?];
                        break :blk @intCast(u16, got.p_vaddr + func.owner_decl.link.elf.offset_table_index * 2);
                    } else if (self.bin_file.cast(link.File.Coff)) |coff_file|
                        @intCast(u16, coff_file.offset_table_virtual_address + func.owner_decl.link.coff.offset_table_index * 2)
                    else
                        unreachable;

                    const return_type = func.owner_decl.typed_value.most_recent.typed_value.ty.fnReturnType();
                    // First, push the return address, then jump; if noreturn, don't bother with the first step
                    // TODO: implement packed struct -> u16 at comptime and move the bitcast here
                    var instr = Instruction{ .condition = .always, .input0 = .immediate, .input1 = .zero, .modify_flags = false, .output = .jump, .command = .load16 };
                    if (return_type.zigTypeTag() == .NoReturn) {
                        try self.code.resize(self.code.items.len + 4);
                        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 4 ..][0..2], @bitCast(u16, instr));
                        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 2 ..][0..2], got_addr);
                        return MCValue.unreach;
                    } else {
                        try self.code.resize(self.code.items.len + 8);
                        var push = Instruction{ .condition = .always, .input0 = .immediate, .input1 = .zero, .modify_flags = false, .output = .push, .command = .ipget };
                        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 8 ..][0..2], @bitCast(u16, push));
                        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 6 ..][0..2], @as(u16, 4));
                        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 4 ..][0..2], @bitCast(u16, instr));
                        mem.writeIntLittle(u16, self.code.items[self.code.items.len - 2 ..][0..2], got_addr);
                        switch (return_type.zigTypeTag()) {
                            .Void => return MCValue{ .none = {} },
                            .NoReturn => unreachable,
                            else => return self.fail(inst.base.src, "TODO implement fn call with non-void return value", .{}),
                        }
                    }
                } else if (func_value.castTag(.extern_fn)) |_| {
                    return self.fail(inst.base.src, "TODO implement calling extern functions", .{});
                } else {
                    return self.fail(inst.base.src, "TODO implement calling bitcasted functions", .{});
                }
            } else {
                return self.fail(inst.base.src, "TODO implement calling runtime known function pointer", .{});
            }
        } else if (self.bin_file.cast(link.File.MachO)) |macho_file| {
            unreachable; // unsupported architecture on MachO
        } else {
            unreachable;
        }

        switch (info.return_value) {
            .register => |reg| {
                if (Register.allocIndex(reg) == null) {
                    // Save function return value in a callee saved register
                    return try self.copyToNewRegister(&inst.base, info.return_value);
                }
            },
            else => {},
        }

        return info.return_value;
    }

    fn genRef(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const operand = try self.resolveInst(inst.operand);
        switch (operand) {
            .unreach => unreachable,
            .dead => unreachable,
            .none => return .none,

            .immediate,
            .register,
            .ptr_stack_offset,
            .ptr_embedded_in_code,
            .compare_flags_unsigned,
            .compare_flags_signed,
            => {
                const stack_offset = try self.allocMemPtr(&inst.base);
                try self.genSetStack(inst.base.src, inst.operand.ty, stack_offset, operand);
                return MCValue{ .ptr_stack_offset = stack_offset };
            },

            .stack_offset => |offset| return MCValue{ .ptr_stack_offset = offset },
            .embedded_in_code => |offset| return MCValue{ .ptr_embedded_in_code = offset },
            .memory => |vaddr| return MCValue{ .immediate = vaddr },

            .undef => return self.fail(inst.base.src, "TODO implement ref on an undefined value", .{}),
        }
    }

    fn ret(self: *Self, src: LazySrcLoc, mcv: MCValue) !MCValue {
        const ret_ty = self.fn_type.fnReturnType();
        try self.setRegOrMem(src, ret_ty, self.ret_mcv, mcv);
        return self.fail(src, "TODO implement return for {}", .{self.target.cpu.arch});
    }

    fn genRet(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const operand = try self.resolveInst(inst.operand);
        return self.ret(inst.base.src, operand);
    }

    fn genRetVoid(self: *Self, inst: *ir.Inst.NoOp) !MCValue {
        return self.ret(inst.base.src, .none);
    }

    fn genCmp(self: *Self, inst: *ir.Inst.BinOp, op: math.CompareOperator) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue{ .dead = {} };
        if (inst.lhs.ty.zigTypeTag() == .ErrorSet or inst.rhs.ty.zigTypeTag() == .ErrorSet)
            return self.fail(inst.base.src, "TODO implement cmp for errors", .{});
        return self.fail(inst.base.src, "TODO implement cmp for {}", .{self.target.cpu.arch});
    }

    fn genDbgStmt(self: *Self, inst: *ir.Inst.DbgStmt) !MCValue {
        // TODO when reworking tzir memory layout, rework source locations here as
        // well to be more efficient, as well as support inlined function calls correctly.
        // For now we convert LazySrcLoc to absolute byte offset, to match what the
        // existing codegen code expects.
        try self.dbgAdvancePCAndLine(inst.byte_offset);
        assert(inst.base.isUnused());
        return MCValue.dead;
    }

    fn genCondBr(self: *Self, inst: *ir.Inst.CondBr) !MCValue {
        return self.fail(inst.base.src, "TODO implement condbr {}", .{self.target.cpu.arch});
    }

    fn genIsNull(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.fail(inst.base.src, "TODO implement isnull for {}", .{self.target.cpu.arch});
    }

    fn genIsNullPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.fail(inst.base.src, "TODO load the operand and call genIsNull", .{});
    }

    fn genIsNonNull(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // Here you can specialize this instruction if it makes sense to, otherwise the default
        // will call genIsNull and invert the result.
        return self.fail(inst.base.src, "TODO call genIsNull and invert the result ", .{});
    }

    fn genIsNonNullPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.fail(inst.base.src, "TODO load the operand and call genIsNonNull", .{});
    }

    fn genIsErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.fail(inst.base.src, "TODO implement iserr for {}", .{self.target.cpu.arch});
    }

    fn genIsErrPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.fail(inst.base.src, "TODO load the operand and call genIsErr", .{});
    }

    fn genErrorToInt(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.resolveInst(inst.operand);
    }

    fn genIntToError(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.resolveInst(inst.operand);
    }

    fn genLoop(self: *Self, inst: *ir.Inst.Loop) !MCValue {
        // A loop is a setup to be able to jump back to the beginning.
        const start_index = self.code.items.len;
        try self.genBody(inst.body);
        try self.jump(inst.base.src, start_index);
        return MCValue.unreach;
    }

    /// Send control flow to the `index` of `self.code`.
    fn jump(self: *Self, src: LazySrcLoc, index: usize) !void {
        return self.fail(src, "TODO implement jump for {}", .{self.target.cpu.arch});
    }

    fn genBlock(self: *Self, inst: *ir.Inst.Block) !MCValue {
        inst.codegen = .{
            // A block is a setup to be able to jump to the end.
            .relocs = .{},
            // It also acts as a receptical for break operands.
            // Here we use `MCValue.none` to represent a null value so that the first
            // break instruction will choose a MCValue for the block result and overwrite
            // this field. Following break instructions will use that MCValue to put their
            // block results.
            .mcv = @bitCast(AnyMCValue, MCValue{ .none = {} }),
        };
        defer inst.codegen.relocs.deinit(self.gpa);

        try self.genBody(inst.body);

        for (inst.codegen.relocs.items) |reloc| try self.performReloc(inst.base.src, reloc);

        return @bitCast(MCValue, inst.codegen.mcv);
    }

    fn genSwitch(self: *Self, inst: *ir.Inst.SwitchBr) !MCValue {
        return self.fail(inst.base.src, "TODO genSwitch for {}", .{self.target.cpu.arch});
    }

    fn performReloc(self: *Self, src: LazySrcLoc, reloc: Reloc) !void {
        switch (reloc) {
            .rel32 => |pos| {
                const amt = self.code.items.len - (pos + 4);
                // Here it would be tempting to implement testing for amt == 0 and then elide the
                // jump. However, that will cause a problem because other jumps may assume that they
                // can jump to this code. Or maybe I didn't understand something when I was debugging.
                // It could be worth another look. Anyway, that's why that isn't done here. Probably the
                // best place to elide jumps will be in semantic analysis, by inlining blocks that only
                // only have 1 break instruction.
                const s32_amt = math.cast(i32, amt) catch
                    return self.fail(src, "unable to perform relocation: jump too far", .{});
                mem.writeIntLittle(i32, self.code.items[pos..][0..4], s32_amt);
            },
            .arm_branch => |info| {
                unreachable; // attempting to perfrom an ARM relocation on a non-ARM target arch
            },
        }
    }

    fn genBrBlockFlat(self: *Self, inst: *ir.Inst.BrBlockFlat) !MCValue {
        try self.genBody(inst.body);
        const last = inst.body.instructions[inst.body.instructions.len - 1];
        return self.br(inst.base.src, inst.block, last);
    }

    fn genBr(self: *Self, inst: *ir.Inst.Br) !MCValue {
        return self.br(inst.base.src, inst.block, inst.operand);
    }

    fn genBrVoid(self: *Self, inst: *ir.Inst.BrVoid) !MCValue {
        return self.brVoid(inst.base.src, inst.block);
    }

    fn genBoolOp(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement boolean operations for {}", .{self.target.cpu.arch});
    }

    fn br(self: *Self, src: LazySrcLoc, block: *ir.Inst.Block, operand: *ir.Inst) !MCValue {
        if (operand.ty.hasCodeGenBits()) {
            const operand_mcv = try self.resolveInst(operand);
            const block_mcv = @bitCast(MCValue, block.codegen.mcv);
            if (block_mcv == .none) {
                block.codegen.mcv = @bitCast(AnyMCValue, operand_mcv);
            } else {
                try self.setRegOrMem(src, block.base.ty, block_mcv, operand_mcv);
            }
        }
        return self.brVoid(src, block);
    }

    fn brVoid(self: *Self, src: LazySrcLoc, block: *ir.Inst.Block) !MCValue {
        // Emit a jump with a relocation. It will be patched up after the block ends.
        try block.codegen.relocs.ensureCapacity(self.gpa, block.codegen.relocs.items.len + 1);

        return self.fail(src, "TODO implement brvoid for {}", .{self.target.cpu.arch});
    }

    fn genAsm(self: *Self, inst: *ir.Inst.Assembly) !MCValue {
        if (!inst.is_volatile and inst.base.isUnused())
            return MCValue.dead;
        if (inst.inputs.len > 0 or inst.output != null) {
            return self.fail(inst.base.src, "TODO implement inline asm inputs / outputs for SPU Mark II", .{});
        }
        if (mem.eql(u8, inst.asm_source, "undefined0")) {
            try self.code.resize(self.code.items.len + 2);
            var instr = Instruction{ .condition = .always, .input0 = .zero, .input1 = .zero, .modify_flags = false, .output = .discard, .command = .undefined0 };
            mem.writeIntLittle(u16, self.code.items[self.code.items.len - 2 ..][0..2], @bitCast(u16, instr));
            return MCValue.none;
        } else {
            return self.fail(inst.base.src, "TODO implement support for more SPU II assembly instructions", .{});
        }
    }

    /// Sets the value without any modifications to register allocation metadata or stack allocation metadata.
    fn setRegOrMem(self: *Self, src: LazySrcLoc, ty: Type, loc: MCValue, val: MCValue) !void {
        switch (loc) {
            .none => return,
            .register => |reg| return self.genSetReg(src, ty, reg, val),
            .stack_offset => |off| return self.genSetStack(src, ty, off, val),
            .memory => {
                return self.fail(src, "TODO implement setRegOrMem for memory", .{});
            },
            else => unreachable,
        }
    }

    fn genSetStack(self: *Self, src: LazySrcLoc, ty: Type, stack_offset: u32, mcv: MCValue) InnerError!void {
        return self.fail(src, "TODO implement getSetStack for {}", .{self.target.cpu.arch});
    }

    fn genSetReg(self: *Self, src: LazySrcLoc, ty: Type, reg: Register, mcv: MCValue) InnerError!void {
        return self.fail(src, "TODO implement getSetReg for {}", .{self.target.cpu.arch});
    }

    fn genPtrToInt(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // no-op
        return self.resolveInst(inst.operand);
    }

    fn genBitCast(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const operand = try self.resolveInst(inst.operand);
        return operand;
    }

    fn resolveInst(self: *Self, inst: *ir.Inst) !MCValue {
        // If the type has no codegen bits, no need to store it.
        if (!inst.ty.hasCodeGenBits())
            return MCValue.none;

        // Constants have static lifetimes, so they are always memoized in the outer most table.
        if (inst.castTag(.constant)) |const_inst| {
            const branch = &self.branch_stack.items[0];
            const gop = try branch.inst_table.getOrPut(self.gpa, inst);
            if (!gop.found_existing) {
                gop.entry.value = try self.genTypedValue(inst.src, .{ .ty = inst.ty, .val = const_inst.val });
            }
            return gop.entry.value;
        }

        return self.getResolvedInstValue(inst);
    }

    fn getResolvedInstValue(self: *Self, inst: *ir.Inst) MCValue {
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

    /// If the MCValue is an immediate, and it does not fit within this type,
    /// we put it in a register.
    /// A potential opportunity for future optimization here would be keeping track
    /// of the fact that the instruction is available both as an immediate
    /// and as a register.
    fn limitImmediateType(self: *Self, inst: *ir.Inst, comptime T: type) !MCValue {
        const mcv = try self.resolveInst(inst);
        const ti = @typeInfo(T).Int;
        switch (mcv) {
            .immediate => |imm| {
                // This immediate is unsigned.
                const U = std.meta.Int(.unsigned, ti.bits - @boolToInt(ti.signedness == .signed));
                if (imm >= math.maxInt(U)) {
                    return MCValue{ .register = try self.copyToTmpRegister(inst.src, Type.initTag(.usize), mcv) };
                }
            },
            else => {},
        }
        return mcv;
    }

    fn genTypedValue(self: *Self, src: LazySrcLoc, typed_value: TypedValue) InnerError!MCValue {
        if (typed_value.val.isUndef())
            return MCValue{ .undef = {} };
        const ptr_bits = self.target.cpu.arch.ptrBitWidth();
        const ptr_bytes: u64 = @divExact(ptr_bits, 8);
        switch (typed_value.ty.zigTypeTag()) {
            .Pointer => {
                if (typed_value.val.castTag(.decl_ref)) |payload| {
                    if (self.bin_file.cast(link.File.Elf)) |elf_file| {
                        const decl = payload.data;
                        const got = &elf_file.program_headers.items[elf_file.phdr_got_index.?];
                        const got_addr = got.p_vaddr + decl.link.elf.offset_table_index * ptr_bytes;
                        return MCValue{ .memory = got_addr };
                    } else if (self.bin_file.cast(link.File.MachO)) |macho_file| {
                        const decl = payload.data;
                        const got_addr = blk: {
                            const seg = macho_file.load_commands.items[macho_file.data_const_segment_cmd_index.?].Segment;
                            const got = seg.sections.items[macho_file.got_section_index.?];
                            break :blk got.addr + decl.link.macho.offset_table_index * ptr_bytes;
                        };
                        return MCValue{ .memory = got_addr };
                    } else if (self.bin_file.cast(link.File.Coff)) |coff_file| {
                        const decl = payload.data;
                        const got_addr = coff_file.offset_table_virtual_address + decl.link.coff.offset_table_index * ptr_bytes;
                        return MCValue{ .memory = got_addr };
                    } else {
                        return self.fail(src, "TODO codegen non-ELF const Decl pointer", .{});
                    }
                }
                return self.fail(src, "TODO codegen more kinds of const pointers", .{});
            },
            .Int => {
                const info = typed_value.ty.intInfo(self.target.*);
                if (info.bits > ptr_bits or info.signedness == .signed) {
                    return self.fail(src, "TODO const int bigger than ptr and signed int", .{});
                }
                return MCValue{ .immediate = typed_value.val.toUnsignedInt() };
            },
            .Bool => {
                return MCValue{ .immediate = @boolToInt(typed_value.val.toBool()) };
            },
            .ComptimeInt => unreachable, // semantic analysis prevents this
            .ComptimeFloat => unreachable, // semantic analysis prevents this
            .Optional => {
                if (typed_value.ty.isPtrLikeOptional()) {
                    if (typed_value.val.isNull())
                        return MCValue{ .immediate = 0 };

                    var buf: Type.Payload.ElemType = undefined;
                    return self.genTypedValue(src, .{
                        .ty = typed_value.ty.optionalChild(&buf),
                        .val = typed_value.val,
                    });
                } else if (typed_value.ty.abiSize(self.target.*) == 1) {
                    return MCValue{ .immediate = @boolToInt(typed_value.val.isNull()) };
                }
                return self.fail(src, "TODO non pointer optionals", .{});
            },
            else => return self.fail(src, "TODO implement const of type '{}'", .{typed_value.ty}),
        }
    }

    const CallMCValues = struct {
        args: []MCValue,
        return_value: MCValue,
        stack_byte_count: u32,
        stack_align: u32,

        fn deinit(self: *CallMCValues, func: *Self) void {
            func.gpa.free(self.args);
            self.* = undefined;
        }
    };

    /// Caller must call `CallMCValues.deinit`.
    fn resolveCallingConventionValues(self: *Self, src: LazySrcLoc, fn_ty: Type) !CallMCValues {
        const cc = fn_ty.fnCallingConvention();
        const param_types = try self.gpa.alloc(Type, fn_ty.fnParamLen());
        defer self.gpa.free(param_types);
        fn_ty.fnParamTypes(param_types);
        var result: CallMCValues = .{
            .args = try self.gpa.alloc(MCValue, param_types.len),
            // These undefined values must be populated before returning from this function.
            .return_value = undefined,
            .stack_byte_count = undefined,
            .stack_align = undefined,
        };
        errdefer self.gpa.free(result.args);

        const ret_ty = fn_ty.fnReturnType();

        if (param_types.len != 0)
            return self.fail(src, "TODO implement codegen parameters for {}", .{self.target.cpu.arch});

        if (ret_ty.zigTypeTag() == .NoReturn) {
            result.return_value = .{ .unreach = {} };
        } else if (!ret_ty.hasCodeGenBits()) {
            result.return_value = .{ .none = {} };
        } else {
            return self.fail(src, "TODO implement codegen return values for {}", .{self.target.cpu.arch});
        }
        return result;
    }

    /// TODO support scope overrides. Also note this logic is duplicated with `Module.wantSafety`.
    fn wantSafety(self: *Self) bool {
        return switch (self.bin_file.options.optimize_mode) {
            .Debug => true,
            .ReleaseSafe => true,
            .ReleaseFast => false,
            .ReleaseSmall => false,
        };
    }

    fn fail(self: *Self, src: LazySrcLoc, comptime format: []const u8, args: anytype) InnerError {
        @setCold(true);
        assert(self.err_msg == null);
        const src_loc = if (src != .unneeded)
            src.toSrcLocWithDecl(self.mod_fn.owner_decl)
        else
            self.src_loc;
        self.err_msg = try ErrorMsg.create(self.bin_file.allocator, src_loc, format, args);
        return error.CodegenFail;
    }

    fn failSymbol(self: *Self, comptime format: []const u8, args: anytype) InnerError {
        @setCold(true);
        assert(self.err_msg == null);
        self.err_msg = try ErrorMsg.create(self.bin_file.allocator, self.src_loc, format, args);
        return error.CodegenFail;
    }

    fn parseRegName(name: []const u8) ?Register {
        if (@hasDecl(Register, "parseRegName")) {
            return Register.parseRegName(name);
        }
        return std.meta.stringToEnum(Register, name);
    }

    fn registerAlias(reg: Register, size_bytes: u32) Register {
        return reg;
    }

    /// For most architectures this does nothing. For x86_64 it resolves any aliased registers
    /// to the 64-bit wide ones.
    fn toCanonicalReg(reg: Register) Register {
        return reg;
    }
};
