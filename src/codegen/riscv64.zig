const std = @import("std");
const DW = std.dwarf;

// TODO: this is only tagged to facilitate the monstrosity.
// Once packed structs work make it packed.
pub const Instruction = union(enum) {
    R: packed struct {
        opcode: u7,
        rd: u5,
        funct3: u3,
        rs1: u5,
        rs2: u5,
        funct7: u7,
    },
    I: packed struct {
        opcode: u7,
        rd: u5,
        funct3: u3,
        rs1: u5,
        imm0_11: u12,
    },
    S: packed struct {
        opcode: u7,
        imm0_4: u5,
        funct3: u3,
        rs1: u5,
        rs2: u5,
        imm5_11: u7,
    },
    B: packed struct {
        opcode: u7,
        imm11: u1,
        imm1_4: u4,
        funct3: u3,
        rs1: u5,
        rs2: u5,
        imm5_10: u6,
        imm12: u1,
    },
    U: packed struct {
        opcode: u7,
        rd: u5,
        imm12_31: u20,
    },
    J: packed struct {
        opcode: u7,
        rd: u5,
        imm12_19: u8,
        imm11: u1,
        imm1_10: u10,
        imm20: u1,
    },

    // TODO: once packed structs work we can remove this monstrosity.
    pub fn toU32(self: Instruction) u32 {
        return switch (self) {
            .R => |v| @bitCast(u32, v),
            .I => |v| @bitCast(u32, v),
            .S => |v| @bitCast(u32, v),
            .B => |v| @intCast(u32, v.opcode) + (@intCast(u32, v.imm11) << 7) + (@intCast(u32, v.imm1_4) << 8) + (@intCast(u32, v.funct3) << 12) + (@intCast(u32, v.rs1) << 15) + (@intCast(u32, v.rs2) << 20) + (@intCast(u32, v.imm5_10) << 25) + (@intCast(u32, v.imm12) << 31),
            .U => |v| @bitCast(u32, v),
            .J => |v| @bitCast(u32, v),
        };
    }

    fn rType(op: u7, fn3: u3, fn7: u7, rd: Register, r1: Register, r2: Register) Instruction {
        return Instruction{
            .R = .{
                .opcode = op,
                .funct3 = fn3,
                .funct7 = fn7,
                .rd = @enumToInt(rd),
                .rs1 = @enumToInt(r1),
                .rs2 = @enumToInt(r2),
            },
        };
    }

    // RISC-V is all signed all the time -- convert immediates to unsigned for processing
    fn iType(op: u7, fn3: u3, rd: Register, r1: Register, imm: i12) Instruction {
        const umm = @bitCast(u12, imm);

        return Instruction{
            .I = .{
                .opcode = op,
                .funct3 = fn3,
                .rd = @enumToInt(rd),
                .rs1 = @enumToInt(r1),
                .imm0_11 = umm,
            },
        };
    }

    fn sType(op: u7, fn3: u3, r1: Register, r2: Register, imm: i12) Instruction {
        const umm = @bitCast(u12, imm);

        return Instruction{
            .S = .{
                .opcode = op,
                .funct3 = fn3,
                .rs1 = @enumToInt(r1),
                .rs2 = @enumToInt(r2),
                .imm0_4 = @truncate(u5, umm),
                .imm5_11 = @truncate(u7, umm >> 5),
            },
        };
    }

    // Use significance value rather than bit value, same for J-type
    // -- less burden on callsite, bonus semantic checking
    fn bType(op: u7, fn3: u3, r1: Register, r2: Register, imm: i13) Instruction {
        const umm = @bitCast(u13, imm);
        if (umm % 2 != 0) @panic("Internal error: misaligned branch target");

        return Instruction{
            .B = .{
                .opcode = op,
                .funct3 = fn3,
                .rs1 = @enumToInt(r1),
                .rs2 = @enumToInt(r2),
                .imm1_4 = @truncate(u4, umm >> 1),
                .imm5_10 = @truncate(u6, umm >> 5),
                .imm11 = @truncate(u1, umm >> 11),
                .imm12 = @truncate(u1, umm >> 12),
            },
        };
    }

    // We have to extract the 20 bits anyway -- let's not make it more painful
    fn uType(op: u7, rd: Register, imm: i20) Instruction {
        const umm = @bitCast(u20, imm);

        return Instruction{
            .U = .{
                .opcode = op,
                .rd = @enumToInt(rd),
                .imm12_31 = umm,
            },
        };
    }

    fn jType(op: u7, rd: Register, imm: i21) Instruction {
        const umm = @bitcast(u21, imm);
        if (umm % 2 != 0) @panic("Internal error: misaligned jump target");

        return Instruction{
            .J = .{
                .opcode = op,
                .rd = @enumToInt(rd),
                .imm1_10 = @truncate(u10, umm >> 1),
                .imm11 = @truncate(u1, umm >> 1),
                .imm12_19 = @truncate(u8, umm >> 12),
                .imm20 = @truncate(u1, umm >> 20),
            },
        };
    }

    // The meat and potatoes. Arguments are in the order in which they would appear in assembly code.

    // Arithmetic/Logical, Register-Register

    pub fn add(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b000, 0b0000000, rd, r1, r2);
    }

    pub fn sub(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b000, 0b0100000, rd, r1, r2);
    }

    pub fn @"and"(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b111, 0b0000000, rd, r1, r2);
    }

    pub fn @"or"(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b110, 0b0000000, rd, r1, r2);
    }

    pub fn xor(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b100, 0b0000000, rd, r1, r2);
    }

    pub fn sll(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b001, 0b0000000, rd, r1, r2);
    }

    pub fn srl(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b101, 0b0000000, rd, r1, r2);
    }

    pub fn sra(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b101, 0b0100000, rd, r1, r2);
    }

    pub fn slt(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b010, 0b0000000, rd, r1, r2);
    }

    pub fn sltu(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0110011, 0b011, 0b0000000, rd, r1, r2);
    }

    // Arithmetic/Logical, Register-Register (32-bit)

    pub fn addw(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0111011, 0b000, rd, r1, r2);
    }

    pub fn subw(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0111011, 0b000, 0b0100000, rd, r1, r2);
    }

    pub fn sllw(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0111011, 0b001, 0b0000000, rd, r1, r2);
    }

    pub fn srlw(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0111011, 0b101, 0b0000000, rd, r1, r2);
    }

    pub fn sraw(rd: Register, r1: Register, r2: Register) Instruction {
        return rType(0b0111011, 0b101, 0b0100000, rd, r1, r2);
    }

    // Arithmetic/Logical, Register-Immediate

    pub fn addi(rd: Register, r1: Register, imm: i12) Instruction {
        return iType(0b0010011, 0b000, rd, r1, imm);
    }

    pub fn andi(rd: Register, r1: Register, imm: i12) Instruction {
        return iType(0b0010011, 0b111, rd, r1, imm);
    }

    pub fn ori(rd: Register, r1: Register, imm: i12) Instruction {
        return iType(0b0010011, 0b110, rd, r1, imm);
    }

    pub fn xori(rd: Register, r1: Register, imm: i12) Instruction {
        return iType(0b0010011, 0b100, rd, r1, imm);
    }

    pub fn slli(rd: Register, r1: Register, shamt: u6) Instruction {
        return iType(0b0010011, 0b001, rd, r1, shamt);
    }

    pub fn srli(rd: Register, r1: Register, shamt: u6) Instruction {
        return iType(0b0010011, 0b101, rd, r1, shamt);
    }

    pub fn srai(rd: Register, r1: Register, shamt: u6) Instruction {
        return iType(0b0010011, 0b101, rd, r1, (1 << 10) + shamt);
    }

    pub fn slti(rd: Register, r1: Register, imm: i12) Instruction {
        return iType(0b0010011, 0b010, rd, r1, imm);
    }

    pub fn sltiu(rd: Register, r1: Register, imm: u12) Instruction {
        return iType(0b0010011, 0b011, rd, r1, @bitCast(i12, imm));
    }

    // Arithmetic/Logical, Register-Immediate (32-bit)

    pub fn addiw(rd: Register, r1: Register, imm: i12) Instruction {
        return iType(0b0011011, 0b000, rd, r1, imm);
    }

    pub fn slliw(rd: Register, r1: Register, shamt: u5) Instruction {
        return iType(0b0011011, 0b001, rd, r1, shamt);
    }

    pub fn srliw(rd: Register, r1: Register, shamt: u5) Instruction {
        return iType(0b0011011, 0b101, rd, r1, shamt);
    }

    pub fn sraiw(rd: Register, r1: Register, shamt: u5) Instruction {
        return iType(0b0011011, 0b101, rd, r1, (1 << 10) + shamt);
    }

    // Upper Immediate

    pub fn lui(rd: Register, imm: i20) Instruction {
        return uType(0b0110111, rd, imm);
    }

    pub fn auipc(rd: Register, imm: i20) Instruction {
        return uType(0b0010111, rd, imm);
    }

    // Load

    pub fn ld(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b011, rd, base, offset);
    }

    pub fn lw(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b010, rd, base, offset);
    }

    pub fn lwu(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b110, rd, base, offset);
    }

    pub fn lh(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b001, rd, base, offset);
    }

    pub fn lhu(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b101, rd, base, offset);
    }

    pub fn lb(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b000, rd, base, offset);
    }

    pub fn lbu(rd: Register, offset: i12, base: Register) Instruction {
        return iType(0b0000011, 0b100, rd, base, offset);
    }

    // Store

    pub fn sd(rs: Register, offset: i12, base: Register) Instruction {
        return sType(0b0100011, 0b011, base, rs, offset);
    }

    pub fn sw(rs: Register, offset: i12, base: Register) Instruction {
        return sType(0b0100011, 0b010, base, rs, offset);
    }

    pub fn sh(rs: Register, offset: i12, base: Register) Instruction {
        return sType(0b0100011, 0b001, base, rs, offset);
    }

    pub fn sb(rs: Register, offset: i12, base: Register) Instruction {
        return sType(0b0100011, 0b000, base, rs, offset);
    }

    // Fence
    // TODO: implement fence

    // Branch

    pub fn beq(r1: Register, r2: Register, offset: u13) Instruction {
        return bType(0b1100011, 0b000, r1, r2, offset);
    }

    pub fn bne(r1: Register, r2: Register, offset: u13) Instruction {
        return bType(0b1100011, 0b001, r1, r2, offset);
    }

    pub fn blt(r1: Register, r2: Register, offset: u13) Instruction {
        return bType(0b1100011, 0b100, r1, r2, offset);
    }

    pub fn bge(r1: Register, r2: Register, offset: u13) Instruction {
        return bType(0b1100011, 0b101, r1, r2, offset);
    }

    pub fn bltu(r1: Register, r2: Register, offset: u13) Instruction {
        return bType(0b1100011, 0b110, r1, r2, offset);
    }

    pub fn bgeu(r1: Register, r2: Register, offset: u13) Instruction {
        return bType(0b1100011, 0b111, r1, r2, offset);
    }

    // Jump

    pub fn jal(link_: Register, offset: i21) Instruction {
        return jType(0b1101111, link_, offset);
    }

    pub fn jalr(link_: Register, offset: i12, base: Register) Instruction {
        return iType(0b1100111, 0b000, link_, base, offset);
    }

    // System

    pub const ecall = iType(0b1110011, 0b000, .zero, .zero, 0x000);
    pub const ebreak = iType(0b1110011, 0b000, .zero, .zero, 0x001);
};

// zig fmt: off
pub const RawRegister = enum(u5) {
    x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7,
    x8,  x9,  x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,

    pub fn dwarfLocOp(reg: RawRegister) u8 {
        return @enumToInt(reg) + DW.OP_reg0;
    }
};

pub const Register = enum(u5) {
    // 64 bit registers
    zero, // zero
    ra, // return address. caller saved
    sp, // stack pointer. callee saved.
    gp, // global pointer
    tp, // thread pointer
    t0, t1, t2, // temporaries. caller saved.
    s0, // s0/fp, callee saved.
    s1, // callee saved.
    a0, a1, // fn args/return values. caller saved.
    a2, a3, a4, a5, a6, a7, // fn args. caller saved.
    s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, // saved registers. callee saved.
    t3, t4, t5, t6, // caller saved
    
    pub fn parseRegName(name: []const u8) ?Register {
        if(std.meta.stringToEnum(Register, name)) |reg| return reg;
        if(std.meta.stringToEnum(RawRegister, name)) |rawreg| return @intToEnum(Register, @enumToInt(rawreg));
        return null;
    }

    /// Returns the index into `callee_preserved_regs`.
    pub fn allocIndex(self: Register) ?u4 {
        inline for(callee_preserved_regs) |cpreg, i| {
            if(self == cpreg) return i;
        }
        return null;
    }

    pub fn dwarfLocOp(reg: Register) u8 {
        return @as(u8, @enumToInt(reg)) + DW.OP_reg0;
    }
};

// zig fmt: on

pub const callee_preserved_regs = [_]Register{
    .s0, .s1, .s2, .s3, .s4, .s5, .s6, .s7, .s8, .s9, .s10, .s11,
};

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

const CodegenUtils = @import("utils.zig");

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

    pub const arch = std.Target.Cpu.Arch.riscv64;

    pub fn getRegisterType() type {
        return Register;
    }

    pub const MCValue = union(enum) {
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
        try CodegenUtils.dbgSetPrologueEnd(Self, self);
        try CodegenUtils.genBody(Self, self, self.mod_fn.body);
        try CodegenUtils.dbgSetEpilogueBegin(Self, self);
        // Drop them off at the rbrace.
        try CodegenUtils.dbgAdvancePCAndLine(Self, self, self.rbrace_src);
    }

    pub fn genFuncInst(self: *Self, inst: *ir.Inst) !MCValue {
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

    pub fn spillInstruction(self: *Self, src: LazySrcLoc, reg: Register, inst: *ir.Inst) !void {
        const stack_mcv = try CodegenUtils.allocRegOrMem(Self, self, inst, false);
        const reg_mcv = CodegenUtils.getResolvedInstValue(Self, self, inst);
        assert(reg == toCanonicalReg(reg_mcv.register));
        const branch = &self.branch_stack.items[self.branch_stack.items.len - 1];
        try branch.inst_table.put(self.gpa, inst, stack_mcv);
        try self.genSetStack(src, inst.ty, stack_mcv.stack_offset, reg_mcv);
    }

    fn genAlloc(self: *Self, inst: *ir.Inst.NoOp) !MCValue {
        const stack_offset = try CodegenUtils.allocMemPtr(Self, self, &inst.base);
        return MCValue{ .ptr_stack_offset = stack_offset };
    }

    fn genFloatCast(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement floatCast for {}", .{self.target.cpu.arch});
    }

    fn genIntCast(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        const operand = try CodegenUtils.resolveInst(Self, self, inst.operand);
        const info_a = inst.operand.ty.intInfo(self.target.*);
        const info_b = inst.base.ty.intInfo(self.target.*);
        if (info_a.signedness != info_b.signedness)
            return CodegenUtils.fail(Self, self, inst.base.src, "TODO gen intcast sign safety in semantic analysis", .{});

        if (info_a.bits == info_b.bits)
            return operand;

        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement intCast for {}", .{self.target.cpu.arch});
    }

    fn genNot(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        const operand = try CodegenUtils.resolveInst(Self, self, inst.operand);
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

        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement NOT for {}", .{self.target.cpu.arch});
    }

    fn genAdd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement add for {}", .{self.target.cpu.arch});
    }

    fn genAddWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement addwrap for {}", .{self.target.cpu.arch});
    }

    fn genMul(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement mul for {}", .{self.target.cpu.arch});
    }

    fn genMulWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement mulwrap for {}", .{self.target.cpu.arch});
    }

    fn genDiv(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement div for {}", .{self.target.cpu.arch});
    }

    fn genBitAnd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement bitwise and for {}", .{self.target.cpu.arch});
    }

    fn genBitOr(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement bitwise or for {}", .{self.target.cpu.arch});
    }

    fn genXor(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement xor for {}", .{self.target.cpu.arch});
    }

    fn genOptionalPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement .optional_payload for {}", .{self.target.cpu.arch});
    }

    fn genOptionalPayloadPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement .optional_payload_ptr for {}", .{self.target.cpu.arch});
    }

    fn genUnwrapErrErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement unwrap error union error for {}", .{self.target.cpu.arch});
    }

    fn genUnwrapErrPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement unwrap error union payload for {}", .{self.target.cpu.arch});
    }
    // *(E!T) -> E
    fn genUnwrapErrErrPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement unwrap error union error ptr for {}", .{self.target.cpu.arch});
    }
    // *(E!T) -> *T
    fn genUnwrapErrPayloadPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement unwrap error union payload ptr for {}", .{self.target.cpu.arch});
    }
    fn genWrapOptional(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const optional_ty = inst.base.ty;

        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        // Optional type is just a boolean true
        if (optional_ty.abiSize(self.target.*) == 1)
            return MCValue{ .immediate = 1 };

        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement wrap optional for {}", .{self.target.cpu.arch});
    }

    /// T to E!T
    fn genWrapErrUnionPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement wrap errunion payload for {}", .{self.target.cpu.arch});
    }

    /// E to E!T
    fn genWrapErrUnionErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement wrap errunion error for {}", .{self.target.cpu.arch});
    }
    fn genVarPtr(self: *Self, inst: *ir.Inst.VarPtr) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement varptr for {}", .{self.target.cpu.arch});
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
        const ptr = try CodegenUtils.resolveInst(Self, self, inst.operand);
        const is_volatile = inst.operand.ty.isVolatilePtr();
        if (inst.base.isUnused() and !is_volatile)
            return MCValue.dead;
        const dst_mcv: MCValue = blk: {
            if (self.reuseOperand(&inst.base, 0, ptr)) {
                // The MCValue that holds the pointer can be re-used as the value.
                break :blk ptr;
            } else {
                break :blk try CodegenUtils.allocRegOrMem(Self, self, &inst.base, true);
            }
        };
        switch (ptr) {
            .none => unreachable,
            .undef => unreachable,
            .unreach => unreachable,
            .dead => unreachable,
            .compare_flags_unsigned => unreachable,
            .compare_flags_signed => unreachable,
            .immediate => |imm| try CodegenUtils.setRegOrMem(Self, self, inst.base.src, elem_ty, dst_mcv, .{ .memory = imm }),
            .ptr_stack_offset => |off| try CodegenUtils.setRegOrMem(Self, self, inst.base.src, elem_ty, dst_mcv, .{ .stack_offset = off }),
            .ptr_embedded_in_code => |off| {
                try CodegenUtils.setRegOrMem(Self, self, inst.base.src, elem_ty, dst_mcv, .{ .embedded_in_code = off });
            },
            .embedded_in_code => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement loading from MCValue.embedded_in_code", .{});
            },
            .register => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement loading from MCValue.register", .{});
            },
            .memory => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement loading from MCValue.memory", .{});
            },
            .stack_offset => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement loading from MCValue.stack_offset", .{});
            },
        }
        return dst_mcv;
    }

    fn genStore(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        const ptr = try CodegenUtils.resolveInst(Self, self, inst.lhs);
        const value = try CodegenUtils.resolveInst(Self, self, inst.rhs);
        const elem_ty = inst.rhs.ty;
        switch (ptr) {
            .none => unreachable,
            .undef => unreachable,
            .unreach => unreachable,
            .dead => unreachable,
            .compare_flags_unsigned => unreachable,
            .compare_flags_signed => unreachable,
            .immediate => |imm| {
                try CodegenUtils.setRegOrMem(Self, self, inst.base.src, elem_ty, .{ .memory = imm }, value);
            },
            .ptr_stack_offset => |off| {
                try self.genSetStack(inst.base.src, elem_ty, off, value);
            },
            .ptr_embedded_in_code => |off| {
                try CodegenUtils.setRegOrMem(Self, self, inst.base.src, elem_ty, .{ .embedded_in_code = off }, value);
            },
            .embedded_in_code => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement storing to MCValue.embedded_in_code", .{});
            },
            .register => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement storing to MCValue.register", .{});
            },
            .memory => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement storing to MCValue.memory", .{});
            },
            .stack_offset => {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement storing to MCValue.stack_offset", .{});
            },
        }
        return .none;
    }

    fn genStructFieldPtr(self: *Self, inst: *ir.Inst.StructFieldPtr) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement codegen struct_field_ptr", .{});
    }

    fn genSub(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement sub for {}", .{self.target.cpu.arch});
    }

    fn genSubWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement subwrap for {}", .{self.target.cpu.arch});
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
                        try CodegenUtils.addDbgInfoTypeReloc(Self, self, inst.base.ty); // DW.AT_type,  DW.FORM_ref4
                        dbg_out.dbg_info.appendSliceAssumeCapacity(name_with_null); // DW.AT_name, DW.FORM_string
                    },
                    .none => {},
                }
            },
            .stack_offset => |offset| {
                switch (self.debug_output) {
                    .dwarf => {},
                    .none => {},
                }
            },
            else => {},
        }
    }

    fn genArg(self: *Self, inst: *ir.Inst.Arg) !MCValue {
        const arg_index = self.arg_index;
        self.arg_index += 1;

        if (callee_preserved_regs.len == 0) {
            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement Register enum for {}", .{self.target.cpu.arch});
        }

        const result = self.args[arg_index];
        const mcv = switch (arch) {
            // TODO support stack-only arguments on all target architectures
            else => result,
        };
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
        mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.ebreak.toU32());
        return .none;
    }

    fn genCall(self: *Self, inst: *ir.Inst.Call) !MCValue {
        var info = try self.resolveCallingConventionValues(inst.base.src, inst.func.ty);
        defer info.deinit(self);

        // Due to incremental compilation, how function calls are generated depends
        // on linking.
        if (self.bin_file.tag == link.File.Elf.base_tag or self.bin_file.tag == link.File.Coff.base_tag) {
            if (info.args.len > 0) return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement fn args for {}", .{self.target.cpu.arch});

            if (inst.func.value()) |func_value| {
                if (func_value.castTag(.function)) |func_payload| {
                    const func = func_payload.data;

                    const ptr_bits = self.target.cpu.arch.ptrBitWidth();
                    const ptr_bytes: u64 = @divExact(ptr_bits, 8);
                    const got_addr = if (self.bin_file.cast(link.File.Elf)) |elf_file| blk: {
                        const got = &elf_file.program_headers.items[elf_file.phdr_got_index.?];
                        break :blk @intCast(u32, got.p_vaddr + func.owner_decl.link.elf.offset_table_index * ptr_bytes);
                    } else if (self.bin_file.cast(link.File.Coff)) |coff_file|
                        coff_file.offset_table_virtual_address + func.owner_decl.link.coff.offset_table_index * ptr_bytes
                    else
                        unreachable;

                    try self.genSetReg(inst.base.src, Type.initTag(.usize), .ra, .{ .memory = got_addr });
                    mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.jalr(.ra, 0, .ra).toU32());
                } else if (func_value.castTag(.extern_fn)) |_| {
                    return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling extern functions", .{});
                } else {
                    return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling bitcasted functions", .{});
                }
            } else {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling runtime known function pointer", .{});
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
                    return try CodegenUtils.copyToNewRegister(Self, self, &inst.base, info.return_value);
                }
            },
            else => {},
        }

        return info.return_value;
    }

    fn genRef(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const operand = try CodegenUtils.resolveInst(Self, self, inst.operand);
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
                const stack_offset = try CodegenUtils.allocMemPtr(Self, self, &inst.base);
                try self.genSetStack(inst.base.src, inst.operand.ty, stack_offset, operand);
                return MCValue{ .ptr_stack_offset = stack_offset };
            },

            .stack_offset => |offset| return MCValue{ .ptr_stack_offset = offset },
            .embedded_in_code => |offset| return MCValue{ .ptr_embedded_in_code = offset },
            .memory => |vaddr| return MCValue{ .immediate = vaddr },

            .undef => return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement ref on an undefined value", .{}),
        }
    }

    fn ret(self: *Self, src: LazySrcLoc, mcv: MCValue) !MCValue {
        const ret_ty = self.fn_type.fnReturnType();
        try CodegenUtils.setRegOrMem(Self, self, src, ret_ty, self.ret_mcv, mcv);
        mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.jalr(.zero, 0, .ra).toU32());
        return .unreach;
    }

    fn genRet(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const operand = try CodegenUtils.resolveInst(Self, self, inst.operand);
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
            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement cmp for errors", .{});
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement cmp for {}", .{self.target.cpu.arch});
    }

    fn genDbgStmt(self: *Self, inst: *ir.Inst.DbgStmt) !MCValue {
        // TODO when reworking tzir memory layout, rework source locations here as
        // well to be more efficient, as well as support inlined function calls correctly.
        // For now we convert LazySrcLoc to absolute byte offset, to match what the
        // existing codegen code expects.
        try CodegenUtils.dbgAdvancePCAndLine(Self, self, inst.byte_offset);
        assert(inst.base.isUnused());
        return MCValue.dead;
    }

    fn genCondBr(self: *Self, inst: *ir.Inst.CondBr) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement condbr {}", .{self.target.cpu.arch});
    }

    fn genIsNull(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement isnull for {}", .{self.target.cpu.arch});
    }

    fn genIsNullPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO load the operand and call genIsNull", .{});
    }

    fn genIsNonNull(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // Here you can specialize this instruction if it makes sense to, otherwise the default
        // will call genIsNull and invert the result.
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO call genIsNull and invert the result ", .{});
    }

    fn genIsNonNullPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO load the operand and call genIsNonNull", .{});
    }

    fn genIsErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement iserr for {}", .{self.target.cpu.arch});
    }

    fn genIsErrPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO load the operand and call genIsErr", .{});
    }

    fn genErrorToInt(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.resolveInst(Self, self, inst.operand);
    }

    fn genIntToError(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return CodegenUtils.resolveInst(Self, self, inst.operand);
    }

    fn genLoop(self: *Self, inst: *ir.Inst.Loop) !MCValue {
        // A loop is a setup to be able to jump back to the beginning.
        const start_index = self.code.items.len;
        try CodegenUtils.genBody(Self, self, inst.body);
        try self.jump(inst.base.src, start_index);
        return MCValue.unreach;
    }

    /// Send control flow to the `index` of `self.code`.
    fn jump(self: *Self, src: LazySrcLoc, index: usize) !void {
        return CodegenUtils.fail(Self, self, src, "TODO implement jump for {}", .{self.target.cpu.arch});
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

        try CodegenUtils.genBody(Self, self, inst.body);

        for (inst.codegen.relocs.items) |reloc| try self.performReloc(inst.base.src, reloc);

        return @bitCast(MCValue, inst.codegen.mcv);
    }

    fn genSwitch(self: *Self, inst: *ir.Inst.SwitchBr) !MCValue {
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO genSwitch for {}", .{self.target.cpu.arch});
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
                    return CodegenUtils.fail(Self, self, src, "unable to perform relocation: jump too far", .{});
                mem.writeIntLittle(i32, self.code.items[pos..][0..4], s32_amt);
            },
            .arm_branch => unreachable, // attempting to perfrom an ARM relocation on a non-ARM target arch
        }
    }

    fn genBrBlockFlat(self: *Self, inst: *ir.Inst.BrBlockFlat) !MCValue {
        try CodegenUtils.genBody(Self, self, inst.body);
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
        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement boolean operations for {}", .{self.target.cpu.arch});
    }

    fn br(self: *Self, src: LazySrcLoc, block: *ir.Inst.Block, operand: *ir.Inst) !MCValue {
        if (operand.ty.hasCodeGenBits()) {
            const operand_mcv = try CodegenUtils.resolveInst(Self, self, operand);
            const block_mcv = @bitCast(MCValue, block.codegen.mcv);
            if (block_mcv == .none) {
                block.codegen.mcv = @bitCast(AnyMCValue, operand_mcv);
            } else {
                try CodegenUtils.setRegOrMem(Self, self, src, block.base.ty, block_mcv, operand_mcv);
            }
        }
        return self.brVoid(src, block);
    }

    fn brVoid(self: *Self, src: LazySrcLoc, block: *ir.Inst.Block) !MCValue {
        return CodegenUtils.fail(Self, self, src, "TODO implement brvoid for {}", .{self.target.cpu.arch});
    }

    fn genAsm(self: *Self, inst: *ir.Inst.Assembly) !MCValue {
        if (!inst.is_volatile and inst.base.isUnused())
            return MCValue.dead;
        for (inst.inputs) |input, i| {
            if (input.len < 3 or input[0] != '{' or input[input.len - 1] != '}') {
                return CodegenUtils.fail(Self, self, inst.base.src, "unrecognized asm input constraint: '{s}'", .{input});
            }
            const reg_name = input[1 .. input.len - 1];
            const reg = parseRegName(reg_name) orelse
                return CodegenUtils.fail(Self, self, inst.base.src, "unrecognized register: '{s}'", .{reg_name});

            const arg = inst.args[i];
            const arg_mcv = try CodegenUtils.resolveInst(Self, self, arg);
            try self.register_manager.getRegWithoutTracking(reg);
            try self.genSetReg(inst.base.src, arg.ty, reg, arg_mcv);
        }

        if (mem.eql(u8, inst.asm_source, "ecall")) {
            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.ecall.toU32());
        } else {
            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement support for more riscv64 assembly instructions", .{});
        }

        if (inst.output_name) |output| {
            if (output.len < 4 or output[0] != '=' or output[1] != '{' or output[output.len - 1] != '}') {
                return CodegenUtils.fail(Self, self, inst.base.src, "unrecognized asm output constraint: '{s}'", .{output});
            }
            const reg_name = output[2 .. output.len - 1];
            const reg = parseRegName(reg_name) orelse
                return CodegenUtils.fail(Self, self, inst.base.src, "unrecognized register: '{s}'", .{reg_name});
            return MCValue{ .register = reg };
        } else {
            return MCValue.none;
        }
    }

    pub fn genSetStack(self: *Self, src: LazySrcLoc, ty: Type, stack_offset: u32, mcv: MCValue) InnerError!void {
        return CodegenUtils.fail(Self, self, src, "TODO implement getSetStack for {}", .{self.target.cpu.arch});
    }

    pub fn genSetReg(self: *Self, src: LazySrcLoc, ty: Type, reg: Register, mcv: MCValue) InnerError!void {
        switch (mcv) {
            .dead => unreachable,
            .ptr_stack_offset => unreachable,
            .ptr_embedded_in_code => unreachable,
            .unreach, .none => return, // Nothing to do.
            .undef => {
                if (!self.wantSafety())
                    return; // The already existing value will do just fine.
                // Write the debug undefined value.
                return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaaaaaaaaaaaaaa });
            },
            .immediate => |unsigned_x| {
                const x = @bitCast(i64, unsigned_x);
                if (math.minInt(i12) <= x and x <= math.maxInt(i12)) {
                    mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.addi(reg, .zero, @truncate(i12, x)).toU32());
                    return;
                }
                if (math.minInt(i32) <= x and x <= math.maxInt(i32)) {
                    const lo12 = @truncate(i12, x);
                    const carry: i32 = if (lo12 < 0) 1 else 0;
                    const hi20 = @truncate(i20, (x >> 12) +% carry);

                    // TODO: add test case for 32-bit immediate
                    mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.lui(reg, hi20).toU32());
                    mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.addi(reg, reg, lo12).toU32());
                    return;
                }
                // li rd, immediate
                // "Myriad sequences"
                return CodegenUtils.fail(Self, self, src, "TODO genSetReg 33-64 bit immediates for riscv64", .{}); // glhf
            },
            .memory => |addr| {
                // The value is in memory at a hard-coded address.
                // If the type is a pointer, it means the pointer address is at this memory location.
                try self.genSetReg(src, ty, reg, .{ .immediate = addr });

                mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.ld(reg, 0, reg).toU32());
                // LOAD imm=[i12 offset = 0], rs1 =

                // return CodegenUtils.fail(Self, self, "TODO implement genSetReg memory for riscv64");
            },
            else => return CodegenUtils.fail(Self, self, src, "TODO implement getSetReg for riscv64 {}", .{mcv}),
        }
    }

    fn genPtrToInt(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // no-op
        return CodegenUtils.resolveInst(Self, self, inst.operand);
    }

    fn genBitCast(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const operand = try CodegenUtils.resolveInst(Self, self, inst.operand);
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

        return CodegenUtils.getResolvedInstValue(Self, self, inst);
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
        const mcv = try CodegenUtils.resolveInst(Self, self, inst);
        const ti = @typeInfo(T).Int;
        switch (mcv) {
            .immediate => |imm| {
                // This immediate is unsigned.
                const U = std.meta.Int(.unsigned, ti.bits - @boolToInt(ti.signedness == .signed));
                if (imm >= math.maxInt(U)) {
                    return MCValue{ .register = try CodegenUtils.copyToTmpRegister(Self, self, inst.src, Type.initTag(.usize), mcv) };
                }
            },
            else => {},
        }
        return mcv;
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
            return CodegenUtils.fail(Self, self, src, "TODO implement codegen parameters for {}", .{self.target.cpu.arch});

        if (ret_ty.zigTypeTag() == .NoReturn) {
            result.return_value = .{ .unreach = {} };
        } else if (!ret_ty.hasCodeGenBits()) {
            result.return_value = .{ .none = {} };
        } else {
            return CodegenUtils.fail(Self, self, src, "TODO implement codegen return values for {}", .{self.target.cpu.arch});
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

    fn parseRegName(name: []const u8) ?Register {
        if (@hasDecl(Register, "parseRegName")) {
            return Register.parseRegName(name);
        }
        return std.meta.stringToEnum(Register, name);
    }

    pub fn registerAlias(reg: Register, size_bytes: u32) Register {
        return reg;
    }

    /// For most architectures this does nothing. For x86_64 it resolves any aliased registers
    /// to the 64-bit wide ones.
    pub fn toCanonicalReg(reg: Register) Register {
        return reg;
    }
};
