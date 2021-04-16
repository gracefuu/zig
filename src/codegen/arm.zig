const std = @import("std");
const DW = std.dwarf;
const testing = std.testing;

/// The condition field specifies the flags neccessary for an
/// Instruction to be executed
pub const Condition = enum(u4) {
    /// equal
    eq,
    /// not equal
    ne,
    /// unsigned higher or same
    cs,
    /// unsigned lower
    cc,
    /// negative
    mi,
    /// positive or zero
    pl,
    /// overflow
    vs,
    /// no overflow
    vc,
    /// unsigned higer
    hi,
    /// unsigned lower or same
    ls,
    /// greater or equal
    ge,
    /// less than
    lt,
    /// greater than
    gt,
    /// less than or equal
    le,
    /// always
    al,

    /// Converts a std.math.CompareOperator into a condition flag,
    /// i.e. returns the condition that is true iff the result of the
    /// comparison is true. Assumes signed comparison
    pub fn fromCompareOperatorSigned(op: std.math.CompareOperator) Condition {
        return switch (op) {
            .gte => .ge,
            .gt => .gt,
            .neq => .ne,
            .lt => .lt,
            .lte => .le,
            .eq => .eq,
        };
    }

    /// Converts a std.math.CompareOperator into a condition flag,
    /// i.e. returns the condition that is true iff the result of the
    /// comparison is true. Assumes unsigned comparison
    pub fn fromCompareOperatorUnsigned(op: std.math.CompareOperator) Condition {
        return switch (op) {
            .gte => .cs,
            .gt => .hi,
            .neq => .ne,
            .lt => .cc,
            .lte => .ls,
            .eq => .eq,
        };
    }

    /// Returns the condition which is true iff the given condition is
    /// false (if such a condition exists)
    pub fn negate(cond: Condition) Condition {
        return switch (cond) {
            .eq => .ne,
            .ne => .eq,
            .cs => .cc,
            .cc => .cs,
            .mi => .pl,
            .pl => .mi,
            .vs => .vc,
            .vc => .vs,
            .hi => .ls,
            .ls => .hi,
            .ge => .lt,
            .lt => .ge,
            .gt => .le,
            .le => .gt,
            .al => unreachable,
        };
    }
};

test "condition from CompareOperator" {
    testing.expectEqual(@as(Condition, .eq), Condition.fromCompareOperatorSigned(.eq));
    testing.expectEqual(@as(Condition, .eq), Condition.fromCompareOperatorUnsigned(.eq));

    testing.expectEqual(@as(Condition, .gt), Condition.fromCompareOperatorSigned(.gt));
    testing.expectEqual(@as(Condition, .hi), Condition.fromCompareOperatorUnsigned(.gt));

    testing.expectEqual(@as(Condition, .le), Condition.fromCompareOperatorSigned(.lte));
    testing.expectEqual(@as(Condition, .ls), Condition.fromCompareOperatorUnsigned(.lte));
}

test "negate condition" {
    testing.expectEqual(@as(Condition, .eq), Condition.ne.negate());
    testing.expectEqual(@as(Condition, .ne), Condition.eq.negate());
}

/// Represents a register in the ARM instruction set architecture
pub const Register = enum(u5) {
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    r8,
    r9,
    r10,
    r11,
    r12,
    r13,
    r14,
    r15,

    /// Argument / result / scratch register 1
    a1,
    /// Argument / result / scratch register 2
    a2,
    /// Argument / scratch register 3
    a3,
    /// Argument / scratch register 4
    a4,
    /// Variable-register 1
    v1,
    /// Variable-register 2
    v2,
    /// Variable-register 3
    v3,
    /// Variable-register 4
    v4,
    /// Variable-register 5
    v5,
    /// Platform register
    v6,
    /// Variable-register 7
    v7,
    /// Frame pointer or Variable-register 8
    fp,
    /// Intra-Procedure-call scratch register
    ip,
    /// Stack pointer
    sp,
    /// Link register
    lr,
    /// Program counter
    pc,

    /// Returns the unique 4-bit ID of this register which is used in
    /// the machine code
    pub fn id(self: Register) u4 {
        return @truncate(u4, @enumToInt(self));
    }

    /// Returns the index into `callee_preserved_regs`.
    pub fn allocIndex(self: Register) ?u4 {
        inline for (callee_preserved_regs) |cpreg, i| {
            if (self.id() == cpreg.id()) return i;
        }
        return null;
    }

    pub fn dwarfLocOp(self: Register) u8 {
        return @as(u8, self.id()) + DW.OP_reg0;
    }
};

test "Register.id" {
    testing.expectEqual(@as(u4, 15), Register.r15.id());
    testing.expectEqual(@as(u4, 15), Register.pc.id());
}

/// Program status registers containing flags, mode bits and other
/// vital information
pub const Psr = enum {
    cpsr,
    spsr,
};

pub const callee_preserved_regs = [_]Register{ .r4, .r5, .r6, .r7, .r8, .r10 };
pub const c_abi_int_param_regs = [_]Register{ .r0, .r1, .r2, .r3 };
pub const c_abi_int_return_regs = [_]Register{ .r0, .r1 };

/// Represents an instruction in the ARM instruction set architecture
pub const Instruction = union(enum) {
    DataProcessing: packed struct {
        // Note to self: The order of the fields top-to-bottom is
        // right-to-left in the actual 32-bit int representation
        op2: u12,
        rd: u4,
        rn: u4,
        s: u1,
        opcode: u4,
        i: u1,
        fixed: u2 = 0b00,
        cond: u4,
    },
    Multiply: packed struct {
        rn: u4,
        fixed_1: u4 = 0b1001,
        rm: u4,
        ra: u4,
        rd: u4,
        set_cond: u1,
        accumulate: u1,
        fixed_2: u6 = 0b000000,
        cond: u4,
    },
    MultiplyLong: packed struct {
        rn: u4,
        fixed_1: u4 = 0b1001,
        rm: u4,
        rdlo: u4,
        rdhi: u4,
        set_cond: u1,
        accumulate: u1,
        unsigned: u1,
        fixed_2: u5 = 0b00001,
        cond: u4,
    },
    SingleDataTransfer: packed struct {
        offset: u12,
        rd: u4,
        rn: u4,
        load_store: u1,
        write_back: u1,
        byte_word: u1,
        up_down: u1,
        pre_post: u1,
        imm: u1,
        fixed: u2 = 0b01,
        cond: u4,
    },
    ExtraLoadStore: packed struct {
        imm4l: u4,
        fixed_1: u1 = 0b1,
        op2: u2,
        fixed_2: u1 = 0b1,
        imm4h: u4,
        rt: u4,
        rn: u4,
        o1: u1,
        write_back: u1,
        imm: u1,
        up_down: u1,
        pre_index: u1,
        fixed_3: u3 = 0b000,
        cond: u4,
    },
    BlockDataTransfer: packed struct {
        register_list: u16,
        rn: u4,
        load_store: u1,
        write_back: u1,
        psr_or_user: u1,
        up_down: u1,
        pre_post: u1,
        fixed: u3 = 0b100,
        cond: u4,
    },
    Branch: packed struct {
        offset: u24,
        link: u1,
        fixed: u3 = 0b101,
        cond: u4,
    },
    BranchExchange: packed struct {
        rn: u4,
        fixed_1: u1 = 0b1,
        link: u1,
        fixed_2: u22 = 0b0001_0010_1111_1111_1111_00,
        cond: u4,
    },
    SupervisorCall: packed struct {
        comment: u24,
        fixed: u4 = 0b1111,
        cond: u4,
    },
    Breakpoint: packed struct {
        imm4: u4,
        fixed_1: u4 = 0b0111,
        imm12: u12,
        fixed_2_and_cond: u12 = 0b1110_0001_0010,
    },

    /// Represents the possible operations which can be performed by a
    /// DataProcessing instruction
    const Opcode = enum(u4) {
        // Rd := Op1 AND Op2
        @"and",
        // Rd := Op1 EOR Op2
        eor,
        // Rd := Op1 - Op2
        sub,
        // Rd := Op2 - Op1
        rsb,
        // Rd := Op1 + Op2
        add,
        // Rd := Op1 + Op2 + C
        adc,
        // Rd := Op1 - Op2 + C - 1
        sbc,
        // Rd := Op2 - Op1 + C - 1
        rsc,
        // set condition codes on Op1 AND Op2
        tst,
        // set condition codes on Op1 EOR Op2
        teq,
        // set condition codes on Op1 - Op2
        cmp,
        // set condition codes on Op1 + Op2
        cmn,
        // Rd := Op1 OR Op2
        orr,
        // Rd := Op2
        mov,
        // Rd := Op1 AND NOT Op2
        bic,
        // Rd := NOT Op2
        mvn,
    };

    /// Represents the second operand to a data processing instruction
    /// which can either be content from a register or an immediate
    /// value
    pub const Operand = union(enum) {
        Register: packed struct {
            rm: u4,
            shift: u8,
        },
        Immediate: packed struct {
            imm: u8,
            rotate: u4,
        },

        /// Represents multiple ways a register can be shifted. A
        /// register can be shifted by a specific immediate value or
        /// by the contents of another register
        pub const Shift = union(enum) {
            Immediate: packed struct {
                fixed: u1 = 0b0,
                typ: u2,
                amount: u5,
            },
            Register: packed struct {
                fixed_1: u1 = 0b1,
                typ: u2,
                fixed_2: u1 = 0b0,
                rs: u4,
            },

            pub const Type = enum(u2) {
                logical_left,
                logical_right,
                arithmetic_right,
                rotate_right,
            };

            pub const none = Shift{
                .Immediate = .{
                    .amount = 0,
                    .typ = 0,
                },
            };

            pub fn toU8(self: Shift) u8 {
                return switch (self) {
                    .Register => |v| @bitCast(u8, v),
                    .Immediate => |v| @bitCast(u8, v),
                };
            }

            pub fn reg(rs: Register, typ: Type) Shift {
                return Shift{
                    .Register = .{
                        .rs = rs.id(),
                        .typ = @enumToInt(typ),
                    },
                };
            }

            pub fn imm(amount: u5, typ: Type) Shift {
                return Shift{
                    .Immediate = .{
                        .amount = amount,
                        .typ = @enumToInt(typ),
                    },
                };
            }
        };

        pub fn toU12(self: Operand) u12 {
            return switch (self) {
                .Register => |v| @bitCast(u12, v),
                .Immediate => |v| @bitCast(u12, v),
            };
        }

        pub fn reg(rm: Register, shift: Shift) Operand {
            return Operand{
                .Register = .{
                    .rm = rm.id(),
                    .shift = shift.toU8(),
                },
            };
        }

        pub fn imm(immediate: u8, rotate: u4) Operand {
            return Operand{
                .Immediate = .{
                    .imm = immediate,
                    .rotate = rotate,
                },
            };
        }

        /// Tries to convert an unsigned 32 bit integer into an
        /// immediate operand using rotation. Returns null when there
        /// is no conversion
        pub fn fromU32(x: u32) ?Operand {
            const masks = comptime blk: {
                const base_mask: u32 = std.math.maxInt(u8);
                var result = [_]u32{0} ** 16;
                for (result) |*mask, i| mask.* = std.math.rotr(u32, base_mask, 2 * i);
                break :blk result;
            };

            return for (masks) |mask, i| {
                if (x & mask == x) {
                    break Operand{
                        .Immediate = .{
                            .imm = @intCast(u8, std.math.rotl(u32, x, 2 * i)),
                            .rotate = @intCast(u4, i),
                        },
                    };
                }
            } else null;
        }
    };

    /// Represents the offset operand of a load or store
    /// instruction. Data can be loaded from memory with either an
    /// immediate offset or an offset that is stored in some register.
    pub const Offset = union(enum) {
        Immediate: u12,
        Register: packed struct {
            rm: u4,
            shift: u8,
        },

        pub const none = Offset{
            .Immediate = 0,
        };

        pub fn toU12(self: Offset) u12 {
            return switch (self) {
                .Register => |v| @bitCast(u12, v),
                .Immediate => |v| v,
            };
        }

        pub fn reg(rm: Register, shift: u8) Offset {
            return Offset{
                .Register = .{
                    .rm = rm.id(),
                    .shift = shift,
                },
            };
        }

        pub fn imm(immediate: u12) Offset {
            return Offset{
                .Immediate = immediate,
            };
        }
    };

    /// Represents the offset operand of an extra load or store
    /// instruction.
    pub const ExtraLoadStoreOffset = union(enum) {
        immediate: u8,
        register: u4,

        pub const none = ExtraLoadStoreOffset{
            .immediate = 0,
        };

        pub fn reg(register: Register) ExtraLoadStoreOffset {
            return ExtraLoadStoreOffset{
                .register = register.id(),
            };
        }

        pub fn imm(immediate: u8) ExtraLoadStoreOffset {
            return ExtraLoadStoreOffset{
                .immediate = immediate,
            };
        }
    };

    /// Represents the register list operand to a block data transfer
    /// instruction
    pub const RegisterList = packed struct {
        r0: bool = false,
        r1: bool = false,
        r2: bool = false,
        r3: bool = false,
        r4: bool = false,
        r5: bool = false,
        r6: bool = false,
        r7: bool = false,
        r8: bool = false,
        r9: bool = false,
        r10: bool = false,
        r11: bool = false,
        r12: bool = false,
        r13: bool = false,
        r14: bool = false,
        r15: bool = false,
    };

    pub fn toU32(self: Instruction) u32 {
        return switch (self) {
            .DataProcessing => |v| @bitCast(u32, v),
            .Multiply => |v| @bitCast(u32, v),
            .MultiplyLong => |v| @bitCast(u32, v),
            .SingleDataTransfer => |v| @bitCast(u32, v),
            .ExtraLoadStore => |v| @bitCast(u32, v),
            .BlockDataTransfer => |v| @bitCast(u32, v),
            .Branch => |v| @bitCast(u32, v),
            .BranchExchange => |v| @bitCast(u32, v),
            .SupervisorCall => |v| @bitCast(u32, v),
            .Breakpoint => |v| @intCast(u32, v.imm4) | (@intCast(u32, v.fixed_1) << 4) | (@intCast(u32, v.imm12) << 8) | (@intCast(u32, v.fixed_2_and_cond) << 20),
        };
    }

    // Helper functions for the "real" functions below

    fn dataProcessing(
        cond: Condition,
        opcode: Opcode,
        s: u1,
        rd: Register,
        rn: Register,
        op2: Operand,
    ) Instruction {
        return Instruction{
            .DataProcessing = .{
                .cond = @enumToInt(cond),
                .i = @boolToInt(op2 == .Immediate),
                .opcode = @enumToInt(opcode),
                .s = s,
                .rn = rn.id(),
                .rd = rd.id(),
                .op2 = op2.toU12(),
            },
        };
    }

    fn specialMov(
        cond: Condition,
        rd: Register,
        imm: u16,
        top: bool,
    ) Instruction {
        return Instruction{
            .DataProcessing = .{
                .cond = @enumToInt(cond),
                .i = 1,
                .opcode = if (top) 0b1010 else 0b1000,
                .s = 0,
                .rn = @truncate(u4, imm >> 12),
                .rd = rd.id(),
                .op2 = @truncate(u12, imm),
            },
        };
    }

    fn multiply(
        cond: Condition,
        set_cond: u1,
        rd: Register,
        rn: Register,
        rm: Register,
        ra: ?Register,
    ) Instruction {
        return Instruction{
            .Multiply = .{
                .cond = @enumToInt(cond),
                .accumulate = @boolToInt(ra != null),
                .set_cond = set_cond,
                .rd = rd.id(),
                .rn = rn.id(),
                .ra = if (ra) |reg| reg.id() else 0b0000,
                .rm = rm.id(),
            },
        };
    }

    fn multiplyLong(
        cond: Condition,
        signed: u1,
        accumulate: u1,
        set_cond: u1,
        rdhi: Register,
        rdlo: Register,
        rm: Register,
        rn: Register,
    ) Instruction {
        return Instruction{
            .MultiplyLong = .{
                .cond = @enumToInt(cond),
                .unsigned = signed,
                .accumulate = accumulate,
                .set_cond = set_cond,
                .rdlo = rdlo.id(),
                .rdhi = rdhi.id(),
                .rn = rn.id(),
                .rm = rm.id(),
            },
        };
    }

    fn singleDataTransfer(
        cond: Condition,
        rd: Register,
        rn: Register,
        offset: Offset,
        pre_index: bool,
        positive: bool,
        byte_word: u1,
        write_back: bool,
        load_store: u1,
    ) Instruction {
        return Instruction{
            .SingleDataTransfer = .{
                .cond = @enumToInt(cond),
                .rn = rn.id(),
                .rd = rd.id(),
                .offset = offset.toU12(),
                .load_store = load_store,
                .write_back = @boolToInt(write_back),
                .byte_word = byte_word,
                .up_down = @boolToInt(positive),
                .pre_post = @boolToInt(pre_index),
                .imm = @boolToInt(offset != .Immediate),
            },
        };
    }

    fn extraLoadStore(
        cond: Condition,
        pre_index: bool,
        positive: bool,
        write_back: bool,
        o1: u1,
        op2: u2,
        rn: Register,
        rt: Register,
        offset: ExtraLoadStoreOffset,
    ) Instruction {
        const imm4l: u4 = switch (offset) {
            .immediate => |imm| @truncate(u4, imm),
            .register => |reg| reg,
        };
        const imm4h: u4 = switch (offset) {
            .immediate => |imm| @truncate(u4, imm >> 4),
            .register => |reg| 0b0000,
        };

        return Instruction{
            .ExtraLoadStore = .{
                .imm4l = imm4l,
                .op2 = op2,
                .imm4h = imm4h,
                .rt = rt.id(),
                .rn = rn.id(),
                .o1 = o1,
                .write_back = @boolToInt(write_back),
                .imm = @boolToInt(offset == .immediate),
                .up_down = @boolToInt(positive),
                .pre_index = @boolToInt(pre_index),
                .cond = @enumToInt(cond),
            },
        };
    }

    fn blockDataTransfer(
        cond: Condition,
        rn: Register,
        reg_list: RegisterList,
        pre_post: u1,
        up_down: u1,
        psr_or_user: u1,
        write_back: bool,
        load_store: u1,
    ) Instruction {
        return Instruction{
            .BlockDataTransfer = .{
                .register_list = @bitCast(u16, reg_list),
                .rn = rn.id(),
                .load_store = load_store,
                .write_back = @boolToInt(write_back),
                .psr_or_user = psr_or_user,
                .up_down = up_down,
                .pre_post = pre_post,
                .cond = @enumToInt(cond),
            },
        };
    }

    fn branch(cond: Condition, offset: i26, link_: u1) Instruction {
        return Instruction{
            .Branch = .{
                .cond = @enumToInt(cond),
                .link = link_,
                .offset = @bitCast(u24, @intCast(i24, offset >> 2)),
            },
        };
    }

    fn branchExchange(cond: Condition, rn: Register, link_: u1) Instruction {
        return Instruction{
            .BranchExchange = .{
                .cond = @enumToInt(cond),
                .link = link_,
                .rn = rn.id(),
            },
        };
    }

    fn supervisorCall(cond: Condition, comment: u24) Instruction {
        return Instruction{
            .SupervisorCall = .{
                .cond = @enumToInt(cond),
                .comment = comment,
            },
        };
    }

    fn breakpoint(imm: u16) Instruction {
        return Instruction{
            .Breakpoint = .{
                .imm12 = @truncate(u12, imm >> 4),
                .imm4 = @truncate(u4, imm),
            },
        };
    }

    // Public functions replicating assembler syntax as closely as
    // possible

    // Data processing

    pub fn @"and"(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .@"and", 0, rd, rn, op2);
    }

    pub fn ands(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .@"and", 1, rd, rn, op2);
    }

    pub fn eor(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .eor, 0, rd, rn, op2);
    }

    pub fn eors(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .eor, 1, rd, rn, op2);
    }

    pub fn sub(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .sub, 0, rd, rn, op2);
    }

    pub fn subs(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .sub, 1, rd, rn, op2);
    }

    pub fn rsb(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .rsb, 0, rd, rn, op2);
    }

    pub fn rsbs(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .rsb, 1, rd, rn, op2);
    }

    pub fn add(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .add, 0, rd, rn, op2);
    }

    pub fn adds(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .add, 1, rd, rn, op2);
    }

    pub fn adc(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .adc, 0, rd, rn, op2);
    }

    pub fn adcs(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .adc, 1, rd, rn, op2);
    }

    pub fn sbc(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .sbc, 0, rd, rn, op2);
    }

    pub fn sbcs(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .sbc, 1, rd, rn, op2);
    }

    pub fn rsc(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .rsc, 0, rd, rn, op2);
    }

    pub fn rscs(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .rsc, 1, rd, rn, op2);
    }

    pub fn tst(cond: Condition, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .tst, 1, .r0, rn, op2);
    }

    pub fn teq(cond: Condition, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .teq, 1, .r0, rn, op2);
    }

    pub fn cmp(cond: Condition, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .cmp, 1, .r0, rn, op2);
    }

    pub fn cmn(cond: Condition, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .cmn, 1, .r0, rn, op2);
    }

    pub fn orr(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .orr, 0, rd, rn, op2);
    }

    pub fn orrs(cond: Condition, rd: Register, rn: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .orr, 1, rd, rn, op2);
    }

    pub fn mov(cond: Condition, rd: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .mov, 0, rd, .r0, op2);
    }

    pub fn movs(cond: Condition, rd: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .mov, 1, rd, .r0, op2);
    }

    pub fn bic(cond: Condition, rd: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .bic, 0, rd, rn, op2);
    }

    pub fn bics(cond: Condition, rd: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .bic, 1, rd, rn, op2);
    }

    pub fn mvn(cond: Condition, rd: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .mvn, 0, rd, .r0, op2);
    }

    pub fn mvns(cond: Condition, rd: Register, op2: Operand) Instruction {
        return dataProcessing(cond, .mvn, 1, rd, .r0, op2);
    }

    // movw and movt

    pub fn movw(cond: Condition, rd: Register, imm: u16) Instruction {
        return specialMov(cond, rd, imm, false);
    }

    pub fn movt(cond: Condition, rd: Register, imm: u16) Instruction {
        return specialMov(cond, rd, imm, true);
    }

    // PSR transfer

    pub fn mrs(cond: Condition, rd: Register, psr: Psr) Instruction {
        return Instruction{
            .DataProcessing = .{
                .cond = @enumToInt(cond),
                .i = 0,
                .opcode = if (psr == .spsr) 0b1010 else 0b1000,
                .s = 0,
                .rn = 0b1111,
                .rd = rd.id(),
                .op2 = 0b0000_0000_0000,
            },
        };
    }

    pub fn msr(cond: Condition, psr: Psr, op: Operand) Instruction {
        return Instruction{
            .DataProcessing = .{
                .cond = @enumToInt(cond),
                .i = 0,
                .opcode = if (psr == .spsr) 0b1011 else 0b1001,
                .s = 0,
                .rn = 0b1111,
                .rd = 0b1111,
                .op2 = op.toU12(),
            },
        };
    }

    // Multiply

    pub fn mul(cond: Condition, rd: Register, rn: Register, rm: Register) Instruction {
        return multiply(cond, 0, rd, rn, rm, null);
    }

    pub fn muls(cond: Condition, rd: Register, rn: Register, rm: Register) Instruction {
        return multiply(cond, 1, rd, rn, rm, null);
    }

    pub fn mla(cond: Condition, rd: Register, rn: Register, rm: Register, ra: Register) Instruction {
        return multiply(cond, 0, rd, rn, rm, ra);
    }

    pub fn mlas(cond: Condition, rd: Register, rn: Register, rm: Register, ra: Register) Instruction {
        return multiply(cond, 1, rd, rn, rm, ra);
    }

    // Multiply long

    pub fn umull(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 0, 0, 0, rdhi, rdlo, rm, rn);
    }

    pub fn umulls(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 0, 0, 1, rdhi, rdlo, rm, rn);
    }

    pub fn umlal(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 0, 1, 0, rdhi, rdlo, rm, rn);
    }

    pub fn umlals(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 0, 1, 1, rdhi, rdlo, rm, rn);
    }

    pub fn smull(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 1, 0, 0, rdhi, rdlo, rm, rn);
    }

    pub fn smulls(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 1, 0, 1, rdhi, rdlo, rm, rn);
    }

    pub fn smlal(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 1, 1, 0, rdhi, rdlo, rm, rn);
    }

    pub fn smlals(cond: Condition, rdlo: Register, rdhi: Register, rn: Register, rm: Register) Instruction {
        return multiplyLong(cond, 1, 1, 1, rdhi, rdlo, rm, rn);
    }

    // Single data transfer

    pub const OffsetArgs = struct {
        pre_index: bool = true,
        positive: bool = true,
        offset: Offset,
        write_back: bool = false,
    };

    pub fn ldr(cond: Condition, rd: Register, rn: Register, args: OffsetArgs) Instruction {
        return singleDataTransfer(cond, rd, rn, args.offset, args.pre_index, args.positive, 0, args.write_back, 1);
    }

    pub fn ldrb(cond: Condition, rd: Register, rn: Register, args: OffsetArgs) Instruction {
        return singleDataTransfer(cond, rd, rn, args.offset, args.pre_index, args.positive, 1, args.write_back, 1);
    }

    pub fn str(cond: Condition, rd: Register, rn: Register, args: OffsetArgs) Instruction {
        return singleDataTransfer(cond, rd, rn, args.offset, args.pre_index, args.positive, 0, args.write_back, 0);
    }

    pub fn strb(cond: Condition, rd: Register, rn: Register, args: OffsetArgs) Instruction {
        return singleDataTransfer(cond, rd, rn, args.offset, args.pre_index, args.positive, 1, args.write_back, 0);
    }

    // Extra load/store

    pub const ExtraLoadStoreOffsetArgs = struct {
        pre_index: bool = true,
        positive: bool = true,
        offset: ExtraLoadStoreOffset,
        write_back: bool = false,
    };

    pub fn strh(cond: Condition, rt: Register, rn: Register, args: ExtraLoadStoreOffsetArgs) Instruction {
        return extraLoadStore(cond, args.pre_index, args.positive, args.write_back, 0, 0b01, rn, rt, args.offset);
    }

    pub fn ldrh(cond: Condition, rt: Register, rn: Register, args: ExtraLoadStoreOffsetArgs) Instruction {
        return extraLoadStore(cond, args.pre_index, args.positive, args.write_back, 1, 0b01, rn, rt, args.offset);
    }

    // Block data transfer

    pub fn ldmda(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 0, 0, 0, write_back, 1);
    }

    pub fn ldmdb(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 1, 0, 0, write_back, 1);
    }

    pub fn ldmib(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 1, 1, 0, write_back, 1);
    }

    pub fn ldmia(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 0, 1, 0, write_back, 1);
    }

    pub const ldmfa = ldmda;
    pub const ldmea = ldmdb;
    pub const ldmed = ldmib;
    pub const ldmfd = ldmia;
    pub const ldm = ldmia;

    pub fn stmda(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 0, 0, 0, write_back, 0);
    }

    pub fn stmdb(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 1, 0, 0, write_back, 0);
    }

    pub fn stmib(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 1, 1, 0, write_back, 0);
    }

    pub fn stmia(cond: Condition, rn: Register, write_back: bool, reg_list: RegisterList) Instruction {
        return blockDataTransfer(cond, rn, reg_list, 0, 1, 0, write_back, 0);
    }

    pub const stmed = stmda;
    pub const stmfd = stmdb;
    pub const stmfa = stmib;
    pub const stmea = stmia;
    pub const stm = stmia;

    // Branch

    pub fn b(cond: Condition, offset: i26) Instruction {
        return branch(cond, offset, 0);
    }

    pub fn bl(cond: Condition, offset: i26) Instruction {
        return branch(cond, offset, 1);
    }

    // Branch and exchange

    pub fn bx(cond: Condition, rn: Register) Instruction {
        return branchExchange(cond, rn, 0);
    }

    pub fn blx(cond: Condition, rn: Register) Instruction {
        return branchExchange(cond, rn, 1);
    }

    // Supervisor Call

    pub const swi = svc;

    pub fn svc(cond: Condition, comment: u24) Instruction {
        return supervisorCall(cond, comment);
    }

    // Breakpoint

    pub fn bkpt(imm: u16) Instruction {
        return breakpoint(imm);
    }

    // Aliases

    pub fn nop() Instruction {
        return mov(.al, .r0, Instruction.Operand.reg(.r0, Instruction.Operand.Shift.none));
    }

    pub fn pop(cond: Condition, args: anytype) Instruction {
        if (@typeInfo(@TypeOf(args)) != .Struct) {
            @compileError("Expected tuple or struct argument, found " ++ @typeName(@TypeOf(args)));
        }

        if (args.len < 1) {
            @compileError("Expected at least one register");
        } else if (args.len == 1) {
            const reg = args[0];
            return ldr(cond, reg, .sp, .{
                .pre_index = false,
                .positive = true,
                .offset = Offset.imm(4),
                .write_back = false,
            });
        } else {
            var register_list: u16 = 0;
            inline for (args) |arg| {
                const reg = @as(Register, arg);
                register_list |= @as(u16, 1) << reg.id();
            }
            return ldm(cond, .sp, true, @bitCast(RegisterList, register_list));
        }
    }

    pub fn push(cond: Condition, args: anytype) Instruction {
        if (@typeInfo(@TypeOf(args)) != .Struct) {
            @compileError("Expected tuple or struct argument, found " ++ @typeName(@TypeOf(args)));
        }

        if (args.len < 1) {
            @compileError("Expected at least one register");
        } else if (args.len == 1) {
            const reg = args[0];
            return str(cond, reg, .sp, .{
                .pre_index = true,
                .positive = false,
                .offset = Offset.imm(4),
                .write_back = true,
            });
        } else {
            var register_list: u16 = 0;
            inline for (args) |arg| {
                const reg = @as(Register, arg);
                register_list |= @as(u16, 1) << reg.id();
            }
            return stmdb(cond, .sp, true, @bitCast(RegisterList, register_list));
        }
    }
};

test "serialize instructions" {
    const Testcase = struct {
        inst: Instruction,
        expected: u32,
    };

    const testcases = [_]Testcase{
        .{ // add r0, r0, r0
            .inst = Instruction.add(.al, .r0, .r0, Instruction.Operand.reg(.r0, Instruction.Operand.Shift.none)),
            .expected = 0b1110_00_0_0100_0_0000_0000_00000000_0000,
        },
        .{ // mov r4, r2
            .inst = Instruction.mov(.al, .r4, Instruction.Operand.reg(.r2, Instruction.Operand.Shift.none)),
            .expected = 0b1110_00_0_1101_0_0000_0100_00000000_0010,
        },
        .{ // mov r0, #42
            .inst = Instruction.mov(.al, .r0, Instruction.Operand.imm(42, 0)),
            .expected = 0b1110_00_1_1101_0_0000_0000_0000_00101010,
        },
        .{ // mrs r5, cpsr
            .inst = Instruction.mrs(.al, .r5, .cpsr),
            .expected = 0b1110_00010_0_001111_0101_000000000000,
        },
        .{ // mul r0, r1, r2
            .inst = Instruction.mul(.al, .r0, .r1, .r2),
            .expected = 0b1110_000000_0_0_0000_0000_0010_1001_0001,
        },
        .{ // umlal r0, r1, r5, r6
            .inst = Instruction.umlal(.al, .r0, .r1, .r5, .r6),
            .expected = 0b1110_00001_0_1_0_0001_0000_0110_1001_0101,
        },
        .{ // ldr r0, [r2, #42]
            .inst = Instruction.ldr(.al, .r0, .r2, .{
                .offset = Instruction.Offset.imm(42),
            }),
            .expected = 0b1110_01_0_1_1_0_0_1_0010_0000_000000101010,
        },
        .{ // str r0, [r3]
            .inst = Instruction.str(.al, .r0, .r3, .{
                .offset = Instruction.Offset.none,
            }),
            .expected = 0b1110_01_0_1_1_0_0_0_0011_0000_000000000000,
        },
        .{ // strh r1, [r5]
            .inst = Instruction.strh(.al, .r1, .r5, .{
                .offset = Instruction.ExtraLoadStoreOffset.none,
            }),
            .expected = 0b1110_000_1_1_1_0_0_0101_0001_0000_1011_0000,
        },
        .{ // b #12
            .inst = Instruction.b(.al, 12),
            .expected = 0b1110_101_0_0000_0000_0000_0000_0000_0011,
        },
        .{ // bl #-4
            .inst = Instruction.bl(.al, -4),
            .expected = 0b1110_101_1_1111_1111_1111_1111_1111_1111,
        },
        .{ // bx lr
            .inst = Instruction.bx(.al, .lr),
            .expected = 0b1110_0001_0010_1111_1111_1111_0001_1110,
        },
        .{ // svc #0
            .inst = Instruction.svc(.al, 0),
            .expected = 0b1110_1111_0000_0000_0000_0000_0000_0000,
        },
        .{ // bkpt #42
            .inst = Instruction.bkpt(42),
            .expected = 0b1110_0001_0010_000000000010_0111_1010,
        },
        .{ // stmdb r9, {r0}
            .inst = Instruction.stmdb(.al, .r9, false, .{ .r0 = true }),
            .expected = 0b1110_100_1_0_0_0_0_1001_0000000000000001,
        },
        .{ // ldmea r4!, {r2, r5}
            .inst = Instruction.ldmea(.al, .r4, true, .{ .r2 = true, .r5 = true }),
            .expected = 0b1110_100_1_0_0_1_1_0100_0000000000100100,
        },
    };

    for (testcases) |case| {
        const actual = case.inst.toU32();
        testing.expectEqual(case.expected, actual);
    }
}

test "aliases" {
    const Testcase = struct {
        expected: Instruction,
        actual: Instruction,
    };

    const testcases = [_]Testcase{
        .{ // pop { r6 }
            .actual = Instruction.pop(.al, .{.r6}),
            .expected = Instruction.ldr(.al, .r6, .sp, .{
                .pre_index = false,
                .positive = true,
                .offset = Instruction.Offset.imm(4),
                .write_back = false,
            }),
        },
        .{ // pop { r1, r5 }
            .actual = Instruction.pop(.al, .{ .r1, .r5 }),
            .expected = Instruction.ldm(.al, .sp, true, .{ .r1 = true, .r5 = true }),
        },
        .{ // push { r3 }
            .actual = Instruction.push(.al, .{.r3}),
            .expected = Instruction.str(.al, .r3, .sp, .{
                .pre_index = true,
                .positive = false,
                .offset = Instruction.Offset.imm(4),
                .write_back = true,
            }),
        },
        .{ // push { r0, r2 }
            .actual = Instruction.push(.al, .{ .r0, .r2 }),
            .expected = Instruction.stmdb(.al, .sp, true, .{ .r0 = true, .r2 = true }),
        },
    };

    for (testcases) |case| {
        testing.expectEqual(case.expected.toU32(), case.actual.toU32());
    }
}

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

pub fn Function(comptime arch_: std.Target.Cpu.Arch) type {
    const writeInt = switch (arch_.endian()) {
        .Little => mem.writeIntLittle,
        .Big => mem.writeIntBig,
    };

    return struct {
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

        pub const arch = arch_;

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
            const cc = self.fn_type.fnCallingConvention();
            if (cc != .Naked) {
                // push {fp, lr}
                // mov fp, sp
                // sub sp, sp, #reloc
                const prologue_reloc = self.code.items.len;
                try self.code.resize(prologue_reloc + 12);
                writeInt(u32, self.code.items[prologue_reloc + 4 ..][0..4], Instruction.mov(.al, .fp, Instruction.Operand.reg(.sp, Instruction.Operand.Shift.none)).toU32());

                try CodegenUtils.dbgSetPrologueEnd(Self, self);

                try CodegenUtils.genBody(Self, self, self.mod_fn.body);

                // Backpatch push callee saved regs
                var saved_regs = Instruction.RegisterList{
                    .r11 = true, // fp
                    .r14 = true, // lr
                };
                inline for (callee_preserved_regs) |reg, i| {
                    if (self.register_manager.isRegAllocated(reg)) {
                        @field(saved_regs, @tagName(reg)) = true;
                    }
                }
                writeInt(u32, self.code.items[prologue_reloc..][0..4], Instruction.stmdb(.al, .sp, true, saved_regs).toU32());

                // Backpatch stack offset
                const stack_end = self.max_end_stack;
                const aligned_stack_end = mem.alignForward(stack_end, self.stack_align);
                if (Instruction.Operand.fromU32(@intCast(u32, aligned_stack_end))) |op| {
                    writeInt(u32, self.code.items[prologue_reloc + 8 ..][0..4], Instruction.sub(.al, .sp, .sp, op).toU32());
                } else {
                    return CodegenUtils.failSymbol(Self, self, "TODO ARM: allow larger stacks", .{});
                }

                try CodegenUtils.dbgSetEpilogueBegin(Self, self);

                // exitlude jumps
                if (self.exitlude_jump_relocs.items.len == 1) {
                    // There is only one relocation. Hence,
                    // this relocation must be at the end of
                    // the code. Therefore, we can just delete
                    // the space initially reserved for the
                    // jump
                    self.code.items.len -= 4;
                } else for (self.exitlude_jump_relocs.items) |jmp_reloc| {
                    const amt = @intCast(i32, self.code.items.len) - @intCast(i32, jmp_reloc + 8);
                    if (amt == -4) {
                        // This return is at the end of the
                        // code block. We can't just delete
                        // the space because there may be
                        // other jumps we already relocated to
                        // the address. Instead, insert a nop
                        writeInt(u32, self.code.items[jmp_reloc..][0..4], Instruction.nop().toU32());
                    } else {
                        if (math.cast(i26, amt)) |offset| {
                            writeInt(u32, self.code.items[jmp_reloc..][0..4], Instruction.b(.al, offset).toU32());
                        } else |err| {
                            return CodegenUtils.failSymbol(Self, self, "exitlude jump is too large", .{});
                        }
                    }
                }

                // Epilogue: pop callee saved registers (swap lr with pc in saved_regs)
                saved_regs.r14 = false; // lr
                saved_regs.r15 = true; // pc

                // mov sp, fp
                // pop {fp, pc}
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, .sp, Instruction.Operand.reg(.fp, Instruction.Operand.Shift.none)).toU32());
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.ldm(.al, .sp, true, saved_regs).toU32());
            } else {
                try CodegenUtils.dbgSetPrologueEnd(Self, self);
                try CodegenUtils.genBody(Self, self, self.mod_fn.body);
                try CodegenUtils.dbgSetEpilogueBegin(Self, self);
            }

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

            var imm = ir.Inst.Constant{
                .base = .{
                    .tag = .constant,
                    .deaths = 0,
                    .ty = inst.operand.ty,
                    .src = inst.operand.src,
                },
                .val = Value.initTag(.bool_true),
            };
            return try self.genArmBinOp(&inst.base, inst.operand, &imm.base, .not);
        }

        fn genAdd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
            // No side effects, so if it's unreferenced, do nothing.
            if (inst.base.isUnused())
                return MCValue.dead;

            return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .add);
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
            return try self.genArmMul(&inst.base, inst.lhs, inst.rhs);
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
            return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .bit_and);
        }

        fn genBitOr(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
            // No side effects, so if it's unreferenced, do nothing.
            if (inst.base.isUnused())
                return MCValue.dead;
            return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .bit_or);
        }

        fn genXor(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
            // No side effects, so if it's unreferenced, do nothing.
            if (inst.base.isUnused())
                return MCValue.dead;
            return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .xor);
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
            return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .sub);
        }

        fn genSubWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
            // No side effects, so if it's unreferenced, do nothing.
            if (inst.base.isUnused())
                return MCValue.dead;
            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement subwrap for {}", .{self.target.cpu.arch});
        }

        fn genArmBinOp(self: *Self, inst: *ir.Inst, op_lhs: *ir.Inst, op_rhs: *ir.Inst, op: ir.Inst.Tag) !MCValue {
            const lhs = try CodegenUtils.resolveInst(Self, self, op_lhs);
            const rhs = try CodegenUtils.resolveInst(Self, self, op_rhs);

            // Destination must be a register
            var dst_mcv: MCValue = undefined;
            var lhs_mcv: MCValue = undefined;
            var rhs_mcv: MCValue = undefined;
            if (self.reuseOperand(inst, 0, lhs)) {
                // LHS is the destination
                // RHS is the source
                lhs_mcv = if (lhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, lhs) else lhs;
                rhs_mcv = rhs;
                dst_mcv = lhs_mcv;
            } else if (self.reuseOperand(inst, 1, rhs)) {
                // RHS is the destination
                // LHS is the source
                lhs_mcv = lhs;
                rhs_mcv = if (rhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, rhs) else rhs;
                dst_mcv = rhs_mcv;
            } else {
                // TODO save 1 copy instruction by directly allocating the destination register
                // LHS is the destination
                // RHS is the source
                lhs_mcv = try CodegenUtils.copyToNewRegister(Self, self, inst, lhs);
                rhs_mcv = rhs;
                dst_mcv = lhs_mcv;
            }

            try self.genArmBinOpCode(inst.src, dst_mcv.register, lhs_mcv, rhs_mcv, op);
            return dst_mcv;
        }

        fn genArmBinOpCode(
            self: *Self,
            src: LazySrcLoc,
            dst_reg: Register,
            lhs_mcv: MCValue,
            rhs_mcv: MCValue,
            op: ir.Inst.Tag,
        ) !void {
            assert(lhs_mcv == .register or lhs_mcv == .register);

            const swap_lhs_and_rhs = rhs_mcv == .register and lhs_mcv != .register;
            const op1 = if (swap_lhs_and_rhs) rhs_mcv.register else lhs_mcv.register;
            const op2 = if (swap_lhs_and_rhs) lhs_mcv else rhs_mcv;

            const operand = switch (op2) {
                .none => unreachable,
                .undef => unreachable,
                .dead, .unreach => unreachable,
                .compare_flags_unsigned => unreachable,
                .compare_flags_signed => unreachable,
                .ptr_stack_offset => unreachable,
                .ptr_embedded_in_code => unreachable,
                .immediate => |imm| blk: {
                    if (imm > std.math.maxInt(u32)) return CodegenUtils.fail(Self, self, src, "TODO ARM binary arithmetic immediate larger than u32", .{});

                    // Load immediate into register if it doesn't fit
                    // as an operand
                    break :blk Instruction.Operand.fromU32(@intCast(u32, imm)) orelse
                        Instruction.Operand.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u32), op2), Instruction.Operand.Shift.none);
                },
                .register => |reg| Instruction.Operand.reg(reg, Instruction.Operand.Shift.none),
                .stack_offset,
                .embedded_in_code,
                .memory,
                => Instruction.Operand.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u32), op2), Instruction.Operand.Shift.none),
            };

            switch (op) {
                .add => {
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.add(.al, dst_reg, op1, operand).toU32());
                },
                .sub => {
                    if (swap_lhs_and_rhs) {
                        writeInt(u32, try self.code.addManyAsArray(4), Instruction.rsb(.al, dst_reg, op1, operand).toU32());
                    } else {
                        writeInt(u32, try self.code.addManyAsArray(4), Instruction.sub(.al, dst_reg, op1, operand).toU32());
                    }
                },
                .bool_and, .bit_and => {
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.@"and"(.al, dst_reg, op1, operand).toU32());
                },
                .bool_or, .bit_or => {
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.orr(.al, dst_reg, op1, operand).toU32());
                },
                .not, .xor => {
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.eor(.al, dst_reg, op1, operand).toU32());
                },
                .cmp_eq => {
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.cmp(.al, op1, operand).toU32());
                },
                else => unreachable, // not a binary instruction
            }
        }

        fn genArmMul(self: *Self, inst: *ir.Inst, op_lhs: *ir.Inst, op_rhs: *ir.Inst) !MCValue {
            const lhs = try CodegenUtils.resolveInst(Self, self, op_lhs);
            const rhs = try CodegenUtils.resolveInst(Self, self, op_rhs);

            // Destination must be a register
            // LHS must be a register
            // RHS must be a register
            var dst_mcv: MCValue = undefined;
            var lhs_mcv: MCValue = undefined;
            var rhs_mcv: MCValue = undefined;
            if (self.reuseOperand(inst, 0, lhs)) {
                // LHS is the destination
                lhs_mcv = if (lhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, lhs) else lhs;
                rhs_mcv = if (rhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, rhs) else rhs;
                dst_mcv = lhs_mcv;
            } else if (self.reuseOperand(inst, 1, rhs)) {
                // RHS is the destination
                lhs_mcv = if (lhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, lhs) else lhs;
                rhs_mcv = if (rhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, rhs) else rhs;
                dst_mcv = rhs_mcv;
            } else {
                // TODO save 1 copy instruction by directly allocating the destination register
                // LHS is the destination
                lhs_mcv = try CodegenUtils.copyToNewRegister(Self, self, inst, lhs);
                rhs_mcv = if (rhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst, rhs) else rhs;
                dst_mcv = lhs_mcv;
            }

            writeInt(u32, try self.code.addManyAsArray(4), Instruction.mul(.al, dst_mcv.register, lhs_mcv.register, rhs_mcv.register).toU32());
            return dst_mcv;
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
                        .dwarf => |dbg_out| {
                            const ty = inst.base.ty;
                            const abi_size = math.cast(u32, ty.abiSize(self.target.*)) catch {
                                return CodegenUtils.fail(Self, self, inst.base.src, "type '{}' too big to fit into stack frame", .{ty});
                            };
                            const adjusted_stack_offset = math.negateCast(offset + abi_size) catch {
                                return CodegenUtils.fail(Self, self, inst.base.src, "Stack offset too large for arguments", .{});
                            };

                            try dbg_out.dbg_info.append(link.File.Elf.abbrev_parameter);

                            // Get length of the LEB128 stack offset
                            var counting_writer = std.io.countingWriter(std.io.null_writer);
                            leb128.writeILEB128(counting_writer.writer(), adjusted_stack_offset) catch unreachable;

                            // DW.AT_location, DW.FORM_exprloc
                            // ULEB128 dwarf expression length
                            try leb128.writeULEB128(dbg_out.dbg_info.writer(), counting_writer.bytes_written + 1);
                            try dbg_out.dbg_info.append(DW.OP_breg11);
                            try leb128.writeILEB128(dbg_out.dbg_info.writer(), adjusted_stack_offset);

                            try dbg_out.dbg_info.ensureCapacity(dbg_out.dbg_info.items.len + 5 + name_with_null.len);
                            try CodegenUtils.addDbgInfoTypeReloc(Self, self, inst.base.ty); // DW.AT_type,  DW.FORM_ref4
                            dbg_out.dbg_info.appendSliceAssumeCapacity(name_with_null); // DW.AT_name, DW.FORM_string
                        },
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

            // TODO support stack-only arguments on all target architectures
            const mcv = switch (result) {
                // Copy registers to the stack
                .register => |reg| blk: {
                    const ty = inst.base.ty;
                    const abi_size = math.cast(u32, ty.abiSize(self.target.*)) catch {
                        return CodegenUtils.fail(Self, self, inst.base.src, "type '{}' too big to fit into stack frame", .{ty});
                    };
                    const abi_align = ty.abiAlignment(self.target.*);
                    const stack_offset = try CodegenUtils.allocMem(Self, self, &inst.base, abi_size, abi_align);
                    try self.genSetStack(inst.base.src, ty, stack_offset, MCValue{ .register = reg });

                    break :blk MCValue{ .stack_offset = stack_offset };
                },
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
            writeInt(u32, try self.code.addManyAsArray(4), Instruction.bkpt(0).toU32());
            return .none;
        }

        fn genCall(self: *Self, inst: *ir.Inst.Call) !MCValue {
            var info = try self.resolveCallingConventionValues(inst.base.src, inst.func.ty);
            defer info.deinit(self);

            // Due to incremental compilation, how function calls are generated depends
            // on linking.
            if (self.bin_file.tag == link.File.Elf.base_tag or self.bin_file.tag == link.File.Coff.base_tag) {
                for (info.args) |mc_arg, arg_i| {
                    const arg = inst.args[arg_i];
                    const arg_mcv = try CodegenUtils.resolveInst(Self, self, inst.args[arg_i]);

                    switch (mc_arg) {
                        .none => continue,
                        .undef => unreachable,
                        .immediate => unreachable,
                        .unreach => unreachable,
                        .dead => unreachable,
                        .embedded_in_code => unreachable,
                        .memory => unreachable,
                        .compare_flags_signed => unreachable,
                        .compare_flags_unsigned => unreachable,
                        .register => |reg| {
                            try self.register_manager.getRegWithoutTracking(reg);
                            try self.genSetReg(arg.src, arg.ty, reg, arg_mcv);
                        },
                        .stack_offset => {
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling with parameters in memory", .{});
                        },
                        .ptr_stack_offset => {
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling with MCValue.ptr_stack_offset arg", .{});
                        },
                        .ptr_embedded_in_code => {
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling with MCValue.ptr_embedded_in_code arg", .{});
                        },
                    }
                }

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

                        try self.genSetReg(inst.base.src, Type.initTag(.usize), .lr, .{ .memory = got_addr });

                        // TODO: add Instruction.supportedOn
                        // function for ARM
                        if (Target.arm.featureSetHas(self.target.cpu.features, .has_v5t)) {
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.blx(.al, .lr).toU32());
                        } else {
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, .lr, Instruction.Operand.reg(.pc, Instruction.Operand.Shift.none)).toU32());
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.bx(.al, .lr).toU32());
                        }
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
            // Just add space for an instruction, patch this later
            try self.code.resize(self.code.items.len + 4);
            try self.exitlude_jump_relocs.append(self.gpa, self.code.items.len - 4);
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
            const lhs = try CodegenUtils.resolveInst(Self, self, inst.lhs);
            const rhs = try CodegenUtils.resolveInst(Self, self, inst.rhs);

            const src_mcv = rhs;
            const dst_mcv = if (lhs != .register) try CodegenUtils.copyToNewRegister(Self, self, inst.lhs, lhs) else lhs;

            try self.genArmBinOpCode(inst.base.src, dst_mcv.register, dst_mcv, src_mcv, .cmp_eq);
            const info = inst.lhs.ty.intInfo(self.target.*);
            return switch (info.signedness) {
                .signed => MCValue{ .compare_flags_signed = op },
                .unsigned => MCValue{ .compare_flags_unsigned = op },
            };
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
            const cond = try CodegenUtils.resolveInst(Self, self, inst.condition);

            const reloc: Reloc = reloc: {
                const condition: Condition = switch (cond) {
                    .compare_flags_signed => |cmp_op| blk: {
                        // Here we map to the opposite condition because the jump is to the false branch.
                        const condition = Condition.fromCompareOperatorSigned(cmp_op);
                        break :blk condition.negate();
                    },
                    .compare_flags_unsigned => |cmp_op| blk: {
                        // Here we map to the opposite condition because the jump is to the false branch.
                        const condition = Condition.fromCompareOperatorUnsigned(cmp_op);
                        break :blk condition.negate();
                    },
                    .register => |reg| blk: {
                        // cmp reg, 1
                        // bne ...
                        const op = Instruction.Operand.imm(1, 0);
                        writeInt(u32, try self.code.addManyAsArray(4), Instruction.cmp(.al, reg, op).toU32());
                        break :blk .ne;
                    },
                    else => return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement condbr {} when condition is {s}", .{ self.target.cpu.arch, @tagName(cond) }),
                };

                const reloc = Reloc{
                    .arm_branch = .{
                        .pos = self.code.items.len,
                        .cond = condition,
                    },
                };
                try self.code.resize(self.code.items.len + 4);
                break :reloc reloc;
            };

            // Capture the state of register and stack allocation state so that we can revert to it.
            const parent_next_stack_offset = self.next_stack_offset;
            const parent_free_registers = self.register_manager.free_registers;
            var parent_stack = try self.stack.clone(self.gpa);
            defer parent_stack.deinit(self.gpa);
            var parent_registers = try self.register_manager.registers.clone(self.gpa);
            defer parent_registers.deinit(self.gpa);

            try self.branch_stack.append(.{});

            const then_deaths = inst.thenDeaths();
            try CodegenUtils.ensureProcessDeathCapacity(Self, self, then_deaths.len);
            for (then_deaths) |operand| {
                CodegenUtils.processDeath(Self, self, operand);
            }
            try CodegenUtils.genBody(Self, self, inst.then_body);

            // Revert to the previous register and stack allocation state.

            var saved_then_branch = self.branch_stack.pop();
            defer saved_then_branch.deinit(self.gpa);

            self.register_manager.registers.deinit(self.gpa);
            self.register_manager.registers = parent_registers;
            parent_registers = .{};

            self.stack.deinit(self.gpa);
            self.stack = parent_stack;
            parent_stack = .{};

            self.next_stack_offset = parent_next_stack_offset;
            self.register_manager.free_registers = parent_free_registers;

            try self.performReloc(inst.base.src, reloc);
            const else_branch = self.branch_stack.addOneAssumeCapacity();
            else_branch.* = .{};

            const else_deaths = inst.elseDeaths();
            try CodegenUtils.ensureProcessDeathCapacity(Self, self, else_deaths.len);
            for (else_deaths) |operand| {
                CodegenUtils.processDeath(Self, self, operand);
            }
            try CodegenUtils.genBody(Self, self, inst.else_body);

            // At this point, each branch will possibly have conflicting values for where
            // each instruction is stored. They agree, however, on which instructions are alive/dead.
            // We use the first ("then") branch as canonical, and here emit
            // instructions into the second ("else") branch to make it conform.
            // We continue respect the data structure semantic guarantees of the else_branch so
            // that we can use all the code emitting abstractions. This is why at the bottom we
            // assert that parent_branch.free_registers equals the saved_then_branch.free_registers
            // rather than assigning it.
            const parent_branch = &self.branch_stack.items[self.branch_stack.items.len - 2];
            try parent_branch.inst_table.ensureCapacity(self.gpa, parent_branch.inst_table.items().len +
                else_branch.inst_table.items().len);
            for (else_branch.inst_table.items()) |else_entry| {
                const canon_mcv = if (saved_then_branch.inst_table.swapRemove(else_entry.key)) |then_entry| blk: {
                    // The instruction's MCValue is overridden in both branches.
                    parent_branch.inst_table.putAssumeCapacity(else_entry.key, then_entry.value);
                    if (else_entry.value == .dead) {
                        assert(then_entry.value == .dead);
                        continue;
                    }
                    break :blk then_entry.value;
                } else blk: {
                    if (else_entry.value == .dead)
                        continue;
                    // The instruction is only overridden in the else branch.
                    var i: usize = self.branch_stack.items.len - 2;
                    while (true) {
                        i -= 1; // If this overflows, the question is: why wasn't the instruction marked dead?
                        if (self.branch_stack.items[i].inst_table.get(else_entry.key)) |mcv| {
                            assert(mcv != .dead);
                            break :blk mcv;
                        }
                    }
                };
                log.debug("consolidating else_entry {*} {}=>{}", .{ else_entry.key, else_entry.value, canon_mcv });
                // TODO make sure the destination stack offset / register does not already have something
                // going on there.
                try CodegenUtils.setRegOrMem(Self, self, inst.base.src, else_entry.key.ty, canon_mcv, else_entry.value);
                // TODO track the new register / stack allocation
            }
            try parent_branch.inst_table.ensureCapacity(self.gpa, parent_branch.inst_table.items().len +
                saved_then_branch.inst_table.items().len);
            for (saved_then_branch.inst_table.items()) |then_entry| {
                // We already deleted the items from this table that matched the else_branch.
                // So these are all instructions that are only overridden in the then branch.
                parent_branch.inst_table.putAssumeCapacity(then_entry.key, then_entry.value);
                if (then_entry.value == .dead)
                    continue;
                const parent_mcv = blk: {
                    var i: usize = self.branch_stack.items.len - 2;
                    while (true) {
                        i -= 1;
                        if (self.branch_stack.items[i].inst_table.get(then_entry.key)) |mcv| {
                            assert(mcv != .dead);
                            break :blk mcv;
                        }
                    }
                };
                log.debug("consolidating then_entry {*} {}=>{}", .{ then_entry.key, parent_mcv, then_entry.value });
                // TODO make sure the destination stack offset / register does not already have something
                // going on there.
                try CodegenUtils.setRegOrMem(Self, self, inst.base.src, then_entry.key.ty, parent_mcv, then_entry.value);
                // TODO track the new register / stack allocation
            }

            self.branch_stack.pop().deinit(self.gpa);

            return MCValue.unreach;
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
            if (math.cast(i26, @intCast(i32, index) - @intCast(i32, self.code.items.len + 8))) |delta| {
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.b(.al, delta).toU32());
            } else |err| {
                return CodegenUtils.fail(Self, self, src, "TODO: enable larger branch offset", .{});
            }
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
                .arm_branch => |info| {
                    const amt = @intCast(i32, self.code.items.len) - @intCast(i32, info.pos + 8);
                    if (math.cast(i26, amt)) |delta| {
                        writeInt(u32, self.code.items[info.pos..][0..4], Instruction.b(info.cond, delta).toU32());
                    } else |_| {
                        return CodegenUtils.fail(Self, self, src, "TODO: enable larger branch offset", .{});
                    }
                },
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
            switch (inst.base.tag) {
                .bool_and => return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .bool_and),
                .bool_or => return try self.genArmBinOp(&inst.base, inst.lhs, inst.rhs, .bool_or),
                else => unreachable, // Not a boolean operation
            }
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
            // Emit a jump with a relocation. It will be patched up after the block ends.
            try block.codegen.relocs.ensureCapacity(self.gpa, block.codegen.relocs.items.len + 1);

            try self.code.resize(self.code.items.len + 4);
            block.codegen.relocs.appendAssumeCapacity(.{
                .arm_branch = .{
                    .pos = self.code.items.len - 4,
                    .cond = .al,
                },
            });
            return .none;
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

            if (mem.eql(u8, inst.asm_source, "svc #0")) {
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.svc(.al, 0).toU32());
            } else {
                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement support for more arm assembly instructions", .{});
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
            switch (mcv) {
                .dead => unreachable,
                .ptr_stack_offset => unreachable,
                .ptr_embedded_in_code => unreachable,
                .unreach, .none => return, // Nothing to do.
                .undef => {
                    if (!self.wantSafety())
                        return; // The already existing value will do just fine.
                    // TODO Upgrade this to a memset call when we have that available.
                    switch (ty.abiSize(self.target.*)) {
                        1 => return self.genSetStack(src, ty, stack_offset, .{ .immediate = 0xaa }),
                        2 => return self.genSetStack(src, ty, stack_offset, .{ .immediate = 0xaaaa }),
                        4 => return self.genSetStack(src, ty, stack_offset, .{ .immediate = 0xaaaaaaaa }),
                        8 => return self.genSetStack(src, ty, stack_offset, .{ .immediate = 0xaaaaaaaaaaaaaaaa }),
                        else => return CodegenUtils.fail(Self, self, src, "TODO implement memset", .{}),
                    }
                },
                .compare_flags_unsigned => |op| {
                    return CodegenUtils.fail(Self, self, src, "TODO implement set stack variable with compare flags value (unsigned)", .{});
                },
                .compare_flags_signed => |op| {
                    return CodegenUtils.fail(Self, self, src, "TODO implement set stack variable with compare flags value (signed)", .{});
                },
                .immediate => {
                    const reg = try CodegenUtils.copyToTmpRegister(Self, self, src, ty, mcv);
                    return self.genSetStack(src, ty, stack_offset, MCValue{ .register = reg });
                },
                .embedded_in_code => |code_offset| {
                    return CodegenUtils.fail(Self, self, src, "TODO implement set stack variable from embedded_in_code", .{});
                },
                .register => |reg| {
                    const abi_size = ty.abiSize(self.target.*);
                    const adj_off = stack_offset + abi_size;

                    switch (abi_size) {
                        1, 4 => {
                            const offset = if (math.cast(u12, adj_off)) |imm| blk: {
                                break :blk Instruction.Offset.imm(imm);
                            } else |_| Instruction.Offset.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u32), MCValue{ .immediate = adj_off }), 0);
                            const str = switch (abi_size) {
                                1 => Instruction.strb,
                                4 => Instruction.str,
                                else => unreachable,
                            };

                            writeInt(u32, try self.code.addManyAsArray(4), str(.al, reg, .fp, .{
                                .offset = offset,
                                .positive = false,
                            }).toU32());
                        },
                        2 => {
                            const offset = if (adj_off <= math.maxInt(u8)) blk: {
                                break :blk Instruction.ExtraLoadStoreOffset.imm(@intCast(u8, adj_off));
                            } else Instruction.ExtraLoadStoreOffset.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u32), MCValue{ .immediate = adj_off }));

                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.strh(.al, reg, .fp, .{
                                .offset = offset,
                                .positive = false,
                            }).toU32());
                        },
                        else => return CodegenUtils.fail(Self, self, src, "TODO implement storing other types abi_size={}", .{abi_size}),
                    }
                },
                .memory => |vaddr| {
                    return CodegenUtils.fail(Self, self, src, "TODO implement set stack variable from memory vaddr", .{});
                },
                .stack_offset => |off| {
                    if (stack_offset == off)
                        return; // Copy stack variable to itself; nothing to do.

                    const reg = try CodegenUtils.copyToTmpRegister(Self, self, src, ty, mcv);
                    return self.genSetStack(src, ty, stack_offset, MCValue{ .register = reg });
                },
            }
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
                    return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaaaaaa });
                },
                .compare_flags_unsigned,
                .compare_flags_signed,
                => |op| {
                    const condition = switch (mcv) {
                        .compare_flags_unsigned => Condition.fromCompareOperatorUnsigned(op),
                        .compare_flags_signed => Condition.fromCompareOperatorSigned(op),
                        else => unreachable,
                    };

                    // mov reg, 0
                    // moveq reg, 1
                    const zero = Instruction.Operand.imm(0, 0);
                    const one = Instruction.Operand.imm(1, 0);
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, reg, zero).toU32());
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(condition, reg, one).toU32());
                },
                .immediate => |x| {
                    if (x > math.maxInt(u32)) return CodegenUtils.fail(Self, self, src, "ARM registers are 32-bit wide", .{});

                    if (Instruction.Operand.fromU32(@intCast(u32, x))) |op| {
                        writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, reg, op).toU32());
                    } else if (Instruction.Operand.fromU32(~@intCast(u32, x))) |op| {
                        writeInt(u32, try self.code.addManyAsArray(4), Instruction.mvn(.al, reg, op).toU32());
                    } else if (x <= math.maxInt(u16)) {
                        if (Target.arm.featureSetHas(self.target.cpu.features, .has_v7)) {
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.movw(.al, reg, @intCast(u16, x)).toU32());
                        } else {
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, reg, Instruction.Operand.imm(@truncate(u8, x), 0)).toU32());
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.orr(.al, reg, reg, Instruction.Operand.imm(@truncate(u8, x >> 8), 12)).toU32());
                        }
                    } else {
                        // TODO write constant to code and load
                        // relative to pc
                        if (Target.arm.featureSetHas(self.target.cpu.features, .has_v7)) {
                            // immediate: 0xaaaabbbb
                            // movw reg, #0xbbbb
                            // movt reg, #0xaaaa
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.movw(.al, reg, @truncate(u16, x)).toU32());
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.movt(.al, reg, @truncate(u16, x >> 16)).toU32());
                        } else {
                            // immediate: 0xaabbccdd
                            // mov reg, #0xaa
                            // orr reg, reg, #0xbb, 24
                            // orr reg, reg, #0xcc, 16
                            // orr reg, reg, #0xdd, 8
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, reg, Instruction.Operand.imm(@truncate(u8, x), 0)).toU32());
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.orr(.al, reg, reg, Instruction.Operand.imm(@truncate(u8, x >> 8), 12)).toU32());
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.orr(.al, reg, reg, Instruction.Operand.imm(@truncate(u8, x >> 16), 8)).toU32());
                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.orr(.al, reg, reg, Instruction.Operand.imm(@truncate(u8, x >> 24), 4)).toU32());
                        }
                    }
                },
                .register => |src_reg| {
                    // If the registers are the same, nothing to do.
                    if (src_reg.id() == reg.id())
                        return;

                    // mov reg, src_reg
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.mov(.al, reg, Instruction.Operand.reg(src_reg, Instruction.Operand.Shift.none)).toU32());
                },
                .memory => |addr| {
                    // The value is in memory at a hard-coded address.
                    // If the type is a pointer, it means the pointer address is at this memory location.
                    try self.genSetReg(src, ty, reg, .{ .immediate = addr });
                    writeInt(u32, try self.code.addManyAsArray(4), Instruction.ldr(.al, reg, reg, .{ .offset = Instruction.Offset.none }).toU32());
                },
                .stack_offset => |unadjusted_off| {
                    // TODO: maybe addressing from sp instead of fp
                    const abi_size = ty.abiSize(self.target.*);
                    const adj_off = unadjusted_off + abi_size;

                    switch (abi_size) {
                        1, 4 => {
                            const offset = if (adj_off <= math.maxInt(u12)) blk: {
                                break :blk Instruction.Offset.imm(@intCast(u12, adj_off));
                            } else Instruction.Offset.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u32), MCValue{ .immediate = adj_off }), 0);
                            const ldr = switch (abi_size) {
                                1 => Instruction.ldrb,
                                4 => Instruction.ldr,
                                else => unreachable,
                            };

                            writeInt(u32, try self.code.addManyAsArray(4), ldr(.al, reg, .fp, .{
                                .offset = offset,
                                .positive = false,
                            }).toU32());
                        },
                        2 => {
                            const offset = if (adj_off <= math.maxInt(u8)) blk: {
                                break :blk Instruction.ExtraLoadStoreOffset.imm(@intCast(u8, adj_off));
                            } else Instruction.ExtraLoadStoreOffset.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u32), MCValue{ .immediate = adj_off }));

                            writeInt(u32, try self.code.addManyAsArray(4), Instruction.ldrh(.al, reg, .fp, .{
                                .offset = offset,
                                .positive = false,
                            }).toU32());
                        },
                        else => return CodegenUtils.fail(Self, self, src, "TODO a type of size {} is not allowed in a register", .{abi_size}),
                    }
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement getSetReg for arm {}", .{mcv}),
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

            switch (cc) {
                .Naked => {
                    assert(result.args.len == 0);
                    result.return_value = .{ .unreach = {} };
                    result.stack_byte_count = 0;
                    result.stack_align = 1;
                    return result;
                },
                .Unspecified, .C => {
                    // ARM Procedure Call Standard, Chapter 6.5
                    var ncrn: usize = 0; // Next Core Register Number
                    var nsaa: u32 = 0; // Next stacked argument address

                    for (param_types) |ty, i| {
                        if (ty.abiAlignment(self.target.*) == 8)
                            ncrn = std.mem.alignForwardGeneric(usize, ncrn, 2);

                        const param_size = @intCast(u32, ty.abiSize(self.target.*));
                        if (std.math.divCeil(u32, param_size, 4) catch unreachable <= 4 - ncrn) {
                            if (param_size <= 4) {
                                result.args[i] = .{ .register = c_abi_int_param_regs[ncrn] };
                                ncrn += 1;
                            } else {
                                return CodegenUtils.fail(Self, self, src, "TODO MCValues with multiple registers", .{});
                            }
                        } else if (ncrn < 4 and nsaa == 0) {
                            return CodegenUtils.fail(Self, self, src, "TODO MCValues split between registers and stack", .{});
                        } else {
                            ncrn = 4;
                            if (ty.abiAlignment(self.target.*) == 8)
                                nsaa = std.mem.alignForwardGeneric(u32, nsaa, 8);

                            result.args[i] = .{ .stack_offset = nsaa };
                            nsaa += param_size;
                        }
                    }

                    result.stack_byte_count = nsaa;
                    result.stack_align = 4;
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement function parameters for {} on arm", .{cc}),
            }

            if (ret_ty.zigTypeTag() == .NoReturn) {
                result.return_value = .{ .unreach = {} };
            } else if (!ret_ty.hasCodeGenBits()) {
                result.return_value = .{ .none = {} };
            } else switch (cc) {
                .Naked => unreachable,
                .Unspecified, .C => {
                    const ret_ty_size = @intCast(u32, ret_ty.abiSize(self.target.*));
                    if (ret_ty_size <= 4) {
                        result.return_value = .{ .register = c_abi_int_return_regs[0] };
                    } else {
                        return CodegenUtils.fail(Self, self, src, "TODO support more return types for ARM backend", .{});
                    }
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement function return values for {}", .{cc}),
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
}
