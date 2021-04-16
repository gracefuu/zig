const std = @import("std");
const DW = std.dwarf;
const assert = std.debug.assert;
const testing = std.testing;

// zig fmt: off

/// General purpose registers in the AArch64 instruction set
pub const Register = enum(u6) {
    // 64-bit registers
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, xzr,

    // 32-bit registers
    w0, w1, w2, w3, w4, w5, w6, w7,
    w8, w9, w10, w11, w12, w13, w14, w15,
    w16, w17, w18, w19, w20, w21, w22, w23,
    w24, w25, w26, w27, w28, w29, w30, wzr,

    pub const sp = Register.xzr;

    pub fn id(self: Register) u5 {
        return @truncate(u5, @enumToInt(self));
    }

    /// Returns the bit-width of the register.
    pub fn size(self: Register) u7 {
        return switch (@enumToInt(self)) {
            0...31 => 64,
            32...63 => 32,
        };
    }

    /// Convert from any register to its 64 bit alias.
    pub fn to64(self: Register) Register {
        return @intToEnum(Register, self.id());
    }

    /// Convert from any register to its 32 bit alias.
    pub fn to32(self: Register) Register {
        return @intToEnum(Register, @as(u6, self.id()) + 32);
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

// zig fmt: on

pub const callee_preserved_regs = [_]Register{
    .x19, .x20, .x21, .x22, .x23,
    .x24, .x25, .x26, .x27, .x28,
};

pub const c_abi_int_param_regs = [_]Register{ .x0, .x1, .x2, .x3, .x4, .x5, .x6, .x7 };
pub const c_abi_int_return_regs = [_]Register{ .x0, .x1, .x2, .x3, .x4, .x5, .x6, .x7 };

test "Register.id" {
    testing.expectEqual(@as(u5, 0), Register.x0.id());
    testing.expectEqual(@as(u5, 0), Register.w0.id());

    testing.expectEqual(@as(u5, 31), Register.xzr.id());
    testing.expectEqual(@as(u5, 31), Register.wzr.id());

    testing.expectEqual(@as(u5, 31), Register.sp.id());
    testing.expectEqual(@as(u5, 31), Register.sp.id());
}

test "Register.size" {
    testing.expectEqual(@as(u7, 64), Register.x19.size());
    testing.expectEqual(@as(u7, 32), Register.w3.size());
}

test "Register.to64/to32" {
    testing.expectEqual(Register.x0, Register.w0.to64());
    testing.expectEqual(Register.x0, Register.x0.to64());

    testing.expectEqual(Register.w3, Register.w3.to32());
    testing.expectEqual(Register.w3, Register.x3.to32());
}

// zig fmt: off

/// Scalar floating point registers in the aarch64 instruction set
pub const FloatingPointRegister = enum(u8) {
    // 128-bit registers
    q0, q1, q2, q3, q4, q5, q6, q7,
    q8, q9, q10, q11, q12, q13, q14, q15,
    q16, q17, q18, q19, q20, q21, q22, q23,
    q24, q25, q26, q27, q28, q29, q30, q31,

    // 64-bit registers
    d0, d1, d2, d3, d4, d5, d6, d7,
    d8, d9, d10, d11, d12, d13, d14, d15,
    d16, d17, d18, d19, d20, d21, d22, d23,
    d24, d25, d26, d27, d28, d29, d30, d31,

    // 32-bit registers
    s0, s1, s2, s3, s4, s5, s6, s7,
    s8, s9, s10, s11, s12, s13, s14, s15,
    s16, s17, s18, s19, s20, s21, s22, s23,
    s24, s25, s26, s27, s28, s29, s30, s31,

    // 16-bit registers
    h0, h1, h2, h3, h4, h5, h6, h7,
    h8, h9, h10, h11, h12, h13, h14, h15,
    h16, h17, h18, h19, h20, h21, h22, h23,
    h24, h25, h26, h27, h28, h29, h30, h31,

    // 8-bit registers
    b0, b1, b2, b3, b4, b5, b6, b7,
    b8, b9, b10, b11, b12, b13, b14, b15,
    b16, b17, b18, b19, b20, b21, b22, b23,
    b24, b25, b26, b27, b28, b29, b30, b31,

    pub fn id(self: FloatingPointRegister) u5 {
        return @truncate(u5, @enumToInt(self));
    }

    /// Returns the bit-width of the register.
    pub fn size(self: FloatingPointRegister) u8 {
        return switch (@enumToInt(self)) {
            0...31 => 128,
            32...63 => 64,
            64...95 => 32,
            96...127 => 16,
            128...159 => 8,
            else => unreachable,
        };
    }

    /// Convert from any register to its 128 bit alias.
    pub fn to128(self: FloatingPointRegister) FloatingPointRegister {
        return @intToEnum(FloatingPointRegister, self.id());
    }

    /// Convert from any register to its 64 bit alias.
    pub fn to64(self: FloatingPointRegister) FloatingPointRegister {
        return @intToEnum(FloatingPointRegister, @as(u8, self.id()) + 32);
    }

    /// Convert from any register to its 32 bit alias.
    pub fn to32(self: FloatingPointRegister) FloatingPointRegister {
        return @intToEnum(FloatingPointRegister, @as(u8, self.id()) + 64);
    }

    /// Convert from any register to its 16 bit alias.
    pub fn to16(self: FloatingPointRegister) FloatingPointRegister {
        return @intToEnum(FloatingPointRegister, @as(u8, self.id()) + 96);
    }

    /// Convert from any register to its 8 bit alias.
    pub fn to8(self: FloatingPointRegister) FloatingPointRegister {
        return @intToEnum(FloatingPointRegister, @as(u8, self.id()) + 128);
    }
};

// zig fmt: on

test "FloatingPointRegister.id" {
    testing.expectEqual(@as(u5, 0), FloatingPointRegister.b0.id());
    testing.expectEqual(@as(u5, 0), FloatingPointRegister.h0.id());
    testing.expectEqual(@as(u5, 0), FloatingPointRegister.s0.id());
    testing.expectEqual(@as(u5, 0), FloatingPointRegister.d0.id());
    testing.expectEqual(@as(u5, 0), FloatingPointRegister.q0.id());

    testing.expectEqual(@as(u5, 2), FloatingPointRegister.q2.id());
    testing.expectEqual(@as(u5, 31), FloatingPointRegister.d31.id());
}

test "FloatingPointRegister.size" {
    testing.expectEqual(@as(u8, 128), FloatingPointRegister.q1.size());
    testing.expectEqual(@as(u8, 64), FloatingPointRegister.d2.size());
    testing.expectEqual(@as(u8, 32), FloatingPointRegister.s3.size());
    testing.expectEqual(@as(u8, 16), FloatingPointRegister.h4.size());
    testing.expectEqual(@as(u8, 8), FloatingPointRegister.b5.size());
}

test "FloatingPointRegister.toX" {
    testing.expectEqual(FloatingPointRegister.q1, FloatingPointRegister.q1.to128());
    testing.expectEqual(FloatingPointRegister.q2, FloatingPointRegister.b2.to128());
    testing.expectEqual(FloatingPointRegister.q3, FloatingPointRegister.h3.to128());

    testing.expectEqual(FloatingPointRegister.d0, FloatingPointRegister.q0.to64());
    testing.expectEqual(FloatingPointRegister.s1, FloatingPointRegister.d1.to32());
    testing.expectEqual(FloatingPointRegister.h2, FloatingPointRegister.s2.to16());
    testing.expectEqual(FloatingPointRegister.b3, FloatingPointRegister.h3.to8());
}

/// Represents an instruction in the AArch64 instruction set
pub const Instruction = union(enum) {
    move_wide_immediate: packed struct {
        rd: u5,
        imm16: u16,
        hw: u2,
        fixed: u6 = 0b100101,
        opc: u2,
        sf: u1,
    },
    pc_relative_address: packed struct {
        rd: u5,
        immhi: u19,
        fixed: u5 = 0b10000,
        immlo: u2,
        op: u1,
    },
    load_store_register: packed struct {
        rt: u5,
        rn: u5,
        offset: u12,
        opc: u2,
        op1: u2,
        v: u1,
        fixed: u3 = 0b111,
        size: u2,
    },
    load_store_register_pair: packed struct {
        rt1: u5,
        rn: u5,
        rt2: u5,
        imm7: u7,
        load: u1,
        encoding: u2,
        fixed: u5 = 0b101_0_0,
        opc: u2,
    },
    load_literal: packed struct {
        rt: u5,
        imm19: u19,
        fixed: u6 = 0b011_0_00,
        opc: u2,
    },
    exception_generation: packed struct {
        ll: u2,
        op2: u3,
        imm16: u16,
        opc: u3,
        fixed: u8 = 0b1101_0100,
    },
    unconditional_branch_register: packed struct {
        op4: u5,
        rn: u5,
        op3: u6,
        op2: u5,
        opc: u4,
        fixed: u7 = 0b1101_011,
    },
    unconditional_branch_immediate: packed struct {
        imm26: u26,
        fixed: u5 = 0b00101,
        op: u1,
    },
    no_operation: packed struct {
        fixed: u32 = 0b1101010100_0_00_011_0010_0000_000_11111,
    },
    logical_shifted_register: packed struct {
        rd: u5,
        rn: u5,
        imm6: u6,
        rm: u5,
        n: u1,
        shift: u2,
        fixed: u5 = 0b01010,
        opc: u2,
        sf: u1,
    },
    add_subtract_immediate: packed struct {
        rd: u5,
        rn: u5,
        imm12: u12,
        sh: u1,
        fixed: u6 = 0b100010,
        s: u1,
        op: u1,
        sf: u1,
    },
    conditional_branch: struct {
        cond: u4,
        o0: u1,
        imm19: u19,
        o1: u1,
        fixed: u7 = 0b0101010,
    },
    compare_and_branch: struct {
        rt: u5,
        imm19: u19,
        op: u1,
        fixed: u6 = 0b011010,
        sf: u1,
    },

    pub const Shift = struct {
        shift: ShiftType = .lsl,
        amount: u6 = 0,

        pub const ShiftType = enum(u2) {
            lsl,
            lsr,
            asr,
            ror,
        };

        pub const none = Shift{
            .shift = .lsl,
            .amount = 0,
        };
    };

    pub const Condition = enum(u4) {
        /// Integer: Equal
        /// Floating point: Equal
        eq,
        /// Integer: Not equal
        /// Floating point: Not equal or unordered
        ne,
        /// Integer: Carry set
        /// Floating point: Greater than, equal, or unordered
        cs,
        /// Integer: Carry clear
        /// Floating point: Less than
        cc,
        /// Integer: Minus, negative
        /// Floating point: Less than
        mi,
        /// Integer: Plus, positive or zero
        /// Floating point: Greater than, equal, or unordered
        pl,
        /// Integer: Overflow
        /// Floating point: Unordered
        vs,
        /// Integer: No overflow
        /// Floating point: Ordered
        vc,
        /// Integer: Unsigned higher
        /// Floating point: Greater than, or unordered
        hi,
        /// Integer: Unsigned lower or same
        /// Floating point: Less than or equal
        ls,
        /// Integer: Signed greater than or equal
        /// Floating point: Greater than or equal
        ge,
        /// Integer: Signed less than
        /// Floating point: Less than, or unordered
        lt,
        /// Integer: Signed greater than
        /// Floating point: Greater than
        gt,
        /// Integer: Signed less than or equal
        /// Floating point: Less than, equal, or unordered
        le,
        /// Integer: Always
        /// Floating point: Always
        al,
        /// Integer: Always
        /// Floating point: Always
        nv,
    };

    pub fn toU32(self: Instruction) u32 {
        return switch (self) {
            .move_wide_immediate => |v| @bitCast(u32, v),
            .pc_relative_address => |v| @bitCast(u32, v),
            .load_store_register => |v| @bitCast(u32, v),
            .load_store_register_pair => |v| @bitCast(u32, v),
            .load_literal => |v| @bitCast(u32, v),
            .exception_generation => |v| @bitCast(u32, v),
            .unconditional_branch_register => |v| @bitCast(u32, v),
            .unconditional_branch_immediate => |v| @bitCast(u32, v),
            .no_operation => |v| @bitCast(u32, v),
            .logical_shifted_register => |v| @bitCast(u32, v),
            .add_subtract_immediate => |v| @bitCast(u32, v),
            // TODO once packed structs work, this can be refactored
            .conditional_branch => |v| @as(u32, v.cond) | (@as(u32, v.o0) << 4) | (@as(u32, v.imm19) << 5) | (@as(u32, v.o1) << 24) | (@as(u32, v.fixed) << 25),
            .compare_and_branch => |v| @as(u32, v.rt) | (@as(u32, v.imm19) << 5) | (@as(u32, v.op) << 24) | (@as(u32, v.fixed) << 25) | (@as(u32, v.sf) << 31),
        };
    }

    fn moveWideImmediate(
        opc: u2,
        rd: Register,
        imm16: u16,
        shift: u6,
    ) Instruction {
        switch (rd.size()) {
            32 => {
                assert(shift % 16 == 0 and shift <= 16);
                return Instruction{
                    .move_wide_immediate = .{
                        .rd = rd.id(),
                        .imm16 = imm16,
                        .hw = @intCast(u2, shift / 16),
                        .opc = opc,
                        .sf = 0,
                    },
                };
            },
            64 => {
                assert(shift % 16 == 0 and shift <= 48);
                return Instruction{
                    .move_wide_immediate = .{
                        .rd = rd.id(),
                        .imm16 = imm16,
                        .hw = @intCast(u2, shift / 16),
                        .opc = opc,
                        .sf = 1,
                    },
                };
            },
            else => unreachable, // unexpected register size
        }
    }

    fn pcRelativeAddress(rd: Register, imm21: i21, op: u1) Instruction {
        assert(rd.size() == 64);
        const imm21_u = @bitCast(u21, imm21);
        return Instruction{
            .pc_relative_address = .{
                .rd = rd.id(),
                .immlo = @truncate(u2, imm21_u),
                .immhi = @truncate(u19, imm21_u >> 2),
                .op = op,
            },
        };
    }

    /// Represents the offset operand of a load or store instruction.
    /// Data can be loaded from memory with either an immediate offset
    /// or an offset that is stored in some register.
    pub const LoadStoreOffset = union(enum) {
        Immediate: union(enum) {
            PostIndex: i9,
            PreIndex: i9,
            Unsigned: u12,
        },
        Register: struct {
            rm: u5,
            shift: union(enum) {
                Uxtw: u2,
                Lsl: u2,
                Sxtw: u2,
                Sxtx: u2,
            },
        },

        pub const none = LoadStoreOffset{
            .Immediate = .{ .Unsigned = 0 },
        };

        pub fn toU12(self: LoadStoreOffset) u12 {
            return switch (self) {
                .Immediate => |imm_type| switch (imm_type) {
                    .PostIndex => |v| (@intCast(u12, @bitCast(u9, v)) << 2) + 1,
                    .PreIndex => |v| (@intCast(u12, @bitCast(u9, v)) << 2) + 3,
                    .Unsigned => |v| v,
                },
                .Register => |r| switch (r.shift) {
                    .Uxtw => |v| (@intCast(u12, r.rm) << 6) + (@intCast(u12, v) << 2) + 16 + 2050,
                    .Lsl => |v| (@intCast(u12, r.rm) << 6) + (@intCast(u12, v) << 2) + 24 + 2050,
                    .Sxtw => |v| (@intCast(u12, r.rm) << 6) + (@intCast(u12, v) << 2) + 48 + 2050,
                    .Sxtx => |v| (@intCast(u12, r.rm) << 6) + (@intCast(u12, v) << 2) + 56 + 2050,
                },
            };
        }

        pub fn imm(offset: u12) LoadStoreOffset {
            return .{
                .Immediate = .{ .Unsigned = offset },
            };
        }

        pub fn imm_post_index(offset: i9) LoadStoreOffset {
            return .{
                .Immediate = .{ .PostIndex = offset },
            };
        }

        pub fn imm_pre_index(offset: i9) LoadStoreOffset {
            return .{
                .Immediate = .{ .PreIndex = offset },
            };
        }

        pub fn reg(rm: Register) LoadStoreOffset {
            return .{
                .Register = .{
                    .rm = rm.id(),
                    .shift = .{
                        .Lsl = 0,
                    },
                },
            };
        }

        pub fn reg_uxtw(rm: Register, shift: u2) LoadStoreOffset {
            assert(rm.size() == 32 and (shift == 0 or shift == 2));
            return .{
                .Register = .{
                    .rm = rm.id(),
                    .shift = .{
                        .Uxtw = shift,
                    },
                },
            };
        }

        pub fn reg_lsl(rm: Register, shift: u2) LoadStoreOffset {
            assert(rm.size() == 64 and (shift == 0 or shift == 3));
            return .{
                .Register = .{
                    .rm = rm.id(),
                    .shift = .{
                        .Lsl = shift,
                    },
                },
            };
        }

        pub fn reg_sxtw(rm: Register, shift: u2) LoadStoreOffset {
            assert(rm.size() == 32 and (shift == 0 or shift == 2));
            return .{
                .Register = .{
                    .rm = rm.id(),
                    .shift = .{
                        .Sxtw = shift,
                    },
                },
            };
        }

        pub fn reg_sxtx(rm: Register, shift: u2) LoadStoreOffset {
            assert(rm.size() == 64 and (shift == 0 or shift == 3));
            return .{
                .Register = .{
                    .rm = rm.id(),
                    .shift = .{
                        .Sxtx = shift,
                    },
                },
            };
        }
    };

    /// Which kind of load/store to perform
    const LoadStoreVariant = enum {
        /// 32-bit or 64-bit
        str,
        /// 16-bit, zero-extended
        strh,
        /// 8-bit, zero-extended
        strb,
        /// 32-bit or 64-bit
        ldr,
        /// 16-bit, zero-extended
        ldrh,
        /// 8-bit, zero-extended
        ldrb,
    };

    fn loadStoreRegister(
        rt: Register,
        rn: Register,
        offset: LoadStoreOffset,
        variant: LoadStoreVariant,
    ) Instruction {
        const off = offset.toU12();
        const op1: u2 = blk: {
            switch (offset) {
                .Immediate => |imm| switch (imm) {
                    .Unsigned => break :blk 0b01,
                    else => {},
                },
                else => {},
            }
            break :blk 0b00;
        };
        const opc: u2 = switch (variant) {
            .ldr, .ldrh, .ldrb => 0b01,
            .str, .strh, .strb => 0b00,
        };
        return Instruction{
            .load_store_register = .{
                .rt = rt.id(),
                .rn = rn.id(),
                .offset = off,
                .opc = opc,
                .op1 = op1,
                .v = 0,
                .size = blk: {
                    switch (variant) {
                        .ldr, .str => switch (rt.size()) {
                            32 => break :blk 0b10,
                            64 => break :blk 0b11,
                            else => unreachable, // unexpected register size
                        },
                        .ldrh, .strh => break :blk 0b01,
                        .ldrb, .strb => break :blk 0b00,
                    }
                },
            },
        };
    }

    fn loadStoreRegisterPair(
        rt1: Register,
        rt2: Register,
        rn: Register,
        offset: i9,
        encoding: u2,
        load: bool,
    ) Instruction {
        switch (rt1.size()) {
            32 => {
                assert(-256 <= offset and offset <= 252);
                const imm7 = @truncate(u7, @bitCast(u9, offset >> 2));
                return Instruction{
                    .load_store_register_pair = .{
                        .rt1 = rt1.id(),
                        .rn = rn.id(),
                        .rt2 = rt2.id(),
                        .imm7 = imm7,
                        .load = @boolToInt(load),
                        .encoding = encoding,
                        .opc = 0b00,
                    },
                };
            },
            64 => {
                assert(-512 <= offset and offset <= 504);
                const imm7 = @truncate(u7, @bitCast(u9, offset >> 3));
                return Instruction{
                    .load_store_register_pair = .{
                        .rt1 = rt1.id(),
                        .rn = rn.id(),
                        .rt2 = rt2.id(),
                        .imm7 = imm7,
                        .load = @boolToInt(load),
                        .encoding = encoding,
                        .opc = 0b10,
                    },
                };
            },
            else => unreachable, // unexpected register size
        }
    }

    fn loadLiteral(rt: Register, imm19: u19) Instruction {
        switch (rt.size()) {
            32 => {
                return Instruction{
                    .load_literal = .{
                        .rt = rt.id(),
                        .imm19 = imm19,
                        .opc = 0b00,
                    },
                };
            },
            64 => {
                return Instruction{
                    .load_literal = .{
                        .rt = rt.id(),
                        .imm19 = imm19,
                        .opc = 0b01,
                    },
                };
            },
            else => unreachable, // unexpected register size
        }
    }

    fn exceptionGeneration(
        opc: u3,
        op2: u3,
        ll: u2,
        imm16: u16,
    ) Instruction {
        return Instruction{
            .exception_generation = .{
                .ll = ll,
                .op2 = op2,
                .imm16 = imm16,
                .opc = opc,
            },
        };
    }

    fn unconditionalBranchRegister(
        opc: u4,
        op2: u5,
        op3: u6,
        rn: Register,
        op4: u5,
    ) Instruction {
        assert(rn.size() == 64);

        return Instruction{
            .unconditional_branch_register = .{
                .op4 = op4,
                .rn = rn.id(),
                .op3 = op3,
                .op2 = op2,
                .opc = opc,
            },
        };
    }

    fn unconditionalBranchImmediate(
        op: u1,
        offset: i28,
    ) Instruction {
        return Instruction{
            .unconditional_branch_immediate = .{
                .imm26 = @bitCast(u26, @intCast(i26, offset >> 2)),
                .op = op,
            },
        };
    }

    fn logicalShiftedRegister(
        opc: u2,
        n: u1,
        shift: Shift,
        rd: Register,
        rn: Register,
        rm: Register,
    ) Instruction {
        switch (rd.size()) {
            32 => {
                assert(shift.amount < 32);
                return Instruction{
                    .logical_shifted_register = .{
                        .rd = rd.id(),
                        .rn = rn.id(),
                        .imm6 = shift.amount,
                        .rm = rm.id(),
                        .n = n,
                        .shift = @enumToInt(shift.shift),
                        .opc = opc,
                        .sf = 0b0,
                    },
                };
            },
            64 => {
                return Instruction{
                    .logical_shifted_register = .{
                        .rd = rd.id(),
                        .rn = rn.id(),
                        .imm6 = shift.amount,
                        .rm = rm.id(),
                        .n = n,
                        .shift = @enumToInt(shift.shift),
                        .opc = opc,
                        .sf = 0b1,
                    },
                };
            },
            else => unreachable, // unexpected register size
        }
    }

    fn addSubtractImmediate(
        op: u1,
        s: u1,
        rd: Register,
        rn: Register,
        imm12: u12,
        shift: bool,
    ) Instruction {
        return Instruction{
            .add_subtract_immediate = .{
                .rd = rd.id(),
                .rn = rn.id(),
                .imm12 = imm12,
                .sh = @boolToInt(shift),
                .s = s,
                .op = op,
                .sf = switch (rd.size()) {
                    32 => 0b0,
                    64 => 0b1,
                    else => unreachable, // unexpected register size
                },
            },
        };
    }

    fn conditionalBranch(
        o0: u1,
        o1: u1,
        cond: Condition,
        offset: i21,
    ) Instruction {
        assert(offset & 0b11 == 0b00);
        return Instruction{
            .conditional_branch = .{
                .cond = @enumToInt(cond),
                .o0 = o0,
                .imm19 = @bitCast(u19, @intCast(i19, offset >> 2)),
                .o1 = o1,
            },
        };
    }

    fn compareAndBranch(
        op: u1,
        rt: Register,
        offset: i21,
    ) Instruction {
        assert(offset & 0b11 == 0b00);
        return Instruction{
            .compare_and_branch = .{
                .rt = rt.id(),
                .imm19 = @bitCast(u19, @intCast(i19, offset >> 2)),
                .op = op,
                .sf = switch (rt.size()) {
                    32 => 0b0,
                    64 => 0b1,
                    else => unreachable, // unexpected register size
                },
            },
        };
    }

    // Helper functions for assembly syntax functions

    // Move wide (immediate)

    pub fn movn(rd: Register, imm16: u16, shift: u6) Instruction {
        return moveWideImmediate(0b00, rd, imm16, shift);
    }

    pub fn movz(rd: Register, imm16: u16, shift: u6) Instruction {
        return moveWideImmediate(0b10, rd, imm16, shift);
    }

    pub fn movk(rd: Register, imm16: u16, shift: u6) Instruction {
        return moveWideImmediate(0b11, rd, imm16, shift);
    }

    // PC relative address

    pub fn adr(rd: Register, imm21: i21) Instruction {
        return pcRelativeAddress(rd, imm21, 0b0);
    }

    pub fn adrp(rd: Register, imm21: i21) Instruction {
        return pcRelativeAddress(rd, imm21, 0b1);
    }

    // Load or store register

    pub const LdrArgs = union(enum) {
        register: struct {
            rn: Register,
            offset: LoadStoreOffset = LoadStoreOffset.none,
        },
        literal: u19,
    };

    pub fn ldr(rt: Register, args: LdrArgs) Instruction {
        switch (args) {
            .register => |info| return loadStoreRegister(rt, info.rn, info.offset, .ldr),
            .literal => |literal| return loadLiteral(rt, literal),
        }
    }

    pub fn ldrh(rt: Register, rn: Register, args: StrArgs) Instruction {
        return loadStoreRegister(rt, rn, args.offset, .ldrh);
    }

    pub fn ldrb(rt: Register, rn: Register, args: StrArgs) Instruction {
        return loadStoreRegister(rt, rn, args.offset, .ldrb);
    }

    pub const StrArgs = struct {
        offset: LoadStoreOffset = LoadStoreOffset.none,
    };

    pub fn str(rt: Register, rn: Register, args: StrArgs) Instruction {
        return loadStoreRegister(rt, rn, args.offset, .str);
    }

    pub fn strh(rt: Register, rn: Register, args: StrArgs) Instruction {
        return loadStoreRegister(rt, rn, args.offset, .strh);
    }

    pub fn strb(rt: Register, rn: Register, args: StrArgs) Instruction {
        return loadStoreRegister(rt, rn, args.offset, .strb);
    }

    // Load or store pair of registers

    pub const LoadStorePairOffset = struct {
        encoding: enum(u2) {
            PostIndex = 0b01,
            Signed = 0b10,
            PreIndex = 0b11,
        },
        offset: i9,

        pub fn none() LoadStorePairOffset {
            return .{ .encoding = .Signed, .offset = 0 };
        }

        pub fn post_index(imm: i9) LoadStorePairOffset {
            return .{ .encoding = .PostIndex, .offset = imm };
        }

        pub fn pre_index(imm: i9) LoadStorePairOffset {
            return .{ .encoding = .PreIndex, .offset = imm };
        }

        pub fn signed(imm: i9) LoadStorePairOffset {
            return .{ .encoding = .Signed, .offset = imm };
        }
    };

    pub fn ldp(rt1: Register, rt2: Register, rn: Register, offset: LoadStorePairOffset) Instruction {
        return loadStoreRegisterPair(rt1, rt2, rn, offset.offset, @enumToInt(offset.encoding), true);
    }

    pub fn ldnp(rt1: Register, rt2: Register, rn: Register, offset: i9) Instruction {
        return loadStoreRegisterPair(rt1, rt2, rn, offset, 0, true);
    }

    pub fn stp(rt1: Register, rt2: Register, rn: Register, offset: LoadStorePairOffset) Instruction {
        return loadStoreRegisterPair(rt1, rt2, rn, offset.offset, @enumToInt(offset.encoding), false);
    }

    pub fn stnp(rt1: Register, rt2: Register, rn: Register, offset: i9) Instruction {
        return loadStoreRegisterPair(rt1, rt2, rn, offset, 0, false);
    }

    // Exception generation

    pub fn svc(imm16: u16) Instruction {
        return exceptionGeneration(0b000, 0b000, 0b01, imm16);
    }

    pub fn hvc(imm16: u16) Instruction {
        return exceptionGeneration(0b000, 0b000, 0b10, imm16);
    }

    pub fn smc(imm16: u16) Instruction {
        return exceptionGeneration(0b000, 0b000, 0b11, imm16);
    }

    pub fn brk(imm16: u16) Instruction {
        return exceptionGeneration(0b001, 0b000, 0b00, imm16);
    }

    pub fn hlt(imm16: u16) Instruction {
        return exceptionGeneration(0b010, 0b000, 0b00, imm16);
    }

    // Unconditional branch (register)

    pub fn br(rn: Register) Instruction {
        return unconditionalBranchRegister(0b0000, 0b11111, 0b000000, rn, 0b00000);
    }

    pub fn blr(rn: Register) Instruction {
        return unconditionalBranchRegister(0b0001, 0b11111, 0b000000, rn, 0b00000);
    }

    pub fn ret(rn: ?Register) Instruction {
        return unconditionalBranchRegister(0b0010, 0b11111, 0b000000, rn orelse .x30, 0b00000);
    }

    // Unconditional branch (immediate)

    pub fn b(offset: i28) Instruction {
        return unconditionalBranchImmediate(0, offset);
    }

    pub fn bl(offset: i28) Instruction {
        return unconditionalBranchImmediate(1, offset);
    }

    // Nop

    pub fn nop() Instruction {
        return Instruction{ .no_operation = .{} };
    }

    // Logical (shifted register)

    pub fn @"and"(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b00, 0b0, shift, rd, rn, rm);
    }

    pub fn bic(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b00, 0b1, shift, rd, rn, rm);
    }

    pub fn orr(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b01, 0b0, shift, rd, rn, rm);
    }

    pub fn orn(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b01, 0b1, shift, rd, rn, rm);
    }

    pub fn eor(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b10, 0b0, shift, rd, rn, rm);
    }

    pub fn eon(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b10, 0b1, shift, rd, rn, rm);
    }

    pub fn ands(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b11, 0b0, shift, rd, rn, rm);
    }

    pub fn bics(rd: Register, rn: Register, rm: Register, shift: Shift) Instruction {
        return logicalShiftedRegister(0b11, 0b1, shift, rd, rn, rm);
    }

    // Add/subtract (immediate)

    pub fn add(rd: Register, rn: Register, imm: u12, shift: bool) Instruction {
        return addSubtractImmediate(0b0, 0b0, rd, rn, imm, shift);
    }

    pub fn adds(rd: Register, rn: Register, imm: u12, shift: bool) Instruction {
        return addSubtractImmediate(0b0, 0b1, rd, rn, imm, shift);
    }

    pub fn sub(rd: Register, rn: Register, imm: u12, shift: bool) Instruction {
        return addSubtractImmediate(0b1, 0b0, rd, rn, imm, shift);
    }

    pub fn subs(rd: Register, rn: Register, imm: u12, shift: bool) Instruction {
        return addSubtractImmediate(0b1, 0b1, rd, rn, imm, shift);
    }

    // Conditional branch

    pub fn bCond(cond: Condition, offset: i21) Instruction {
        return conditionalBranch(0b0, 0b0, cond, offset);
    }

    // Compare and branch

    pub fn cbz(rt: Register, offset: i21) Instruction {
        return compareAndBranch(0b0, rt, offset);
    }

    pub fn cbnz(rt: Register, offset: i21) Instruction {
        return compareAndBranch(0b1, rt, offset);
    }
};

test {
    testing.refAllDecls(@This());
}

test "serialize instructions" {
    const Testcase = struct {
        inst: Instruction,
        expected: u32,
    };

    const testcases = [_]Testcase{
        .{ // orr x0, xzr, x1
            .inst = Instruction.orr(.x0, .xzr, .x1, Instruction.Shift.none),
            .expected = 0b1_01_01010_00_0_00001_000000_11111_00000,
        },
        .{ // orn x0, xzr, x1
            .inst = Instruction.orn(.x0, .xzr, .x1, Instruction.Shift.none),
            .expected = 0b1_01_01010_00_1_00001_000000_11111_00000,
        },
        .{ // movz x1, #4
            .inst = Instruction.movz(.x1, 4, 0),
            .expected = 0b1_10_100101_00_0000000000000100_00001,
        },
        .{ // movz x1, #4, lsl 16
            .inst = Instruction.movz(.x1, 4, 16),
            .expected = 0b1_10_100101_01_0000000000000100_00001,
        },
        .{ // movz x1, #4, lsl 32
            .inst = Instruction.movz(.x1, 4, 32),
            .expected = 0b1_10_100101_10_0000000000000100_00001,
        },
        .{ // movz x1, #4, lsl 48
            .inst = Instruction.movz(.x1, 4, 48),
            .expected = 0b1_10_100101_11_0000000000000100_00001,
        },
        .{ // movz w1, #4
            .inst = Instruction.movz(.w1, 4, 0),
            .expected = 0b0_10_100101_00_0000000000000100_00001,
        },
        .{ // movz w1, #4, lsl 16
            .inst = Instruction.movz(.w1, 4, 16),
            .expected = 0b0_10_100101_01_0000000000000100_00001,
        },
        .{ // svc #0
            .inst = Instruction.svc(0),
            .expected = 0b1101_0100_000_0000000000000000_00001,
        },
        .{ // svc #0x80 ; typical on Darwin
            .inst = Instruction.svc(0x80),
            .expected = 0b1101_0100_000_0000000010000000_00001,
        },
        .{ // ret
            .inst = Instruction.ret(null),
            .expected = 0b1101_011_00_10_11111_0000_00_11110_00000,
        },
        .{ // bl #0x10
            .inst = Instruction.bl(0x10),
            .expected = 0b1_00101_00_0000_0000_0000_0000_0000_0100,
        },
        .{ // ldr x2, [x1]
            .inst = Instruction.ldr(.x2, .{ .register = .{ .rn = .x1 } }),
            .expected = 0b11_111_0_01_01_000000000000_00001_00010,
        },
        .{ // ldr x2, [x1, #1]!
            .inst = Instruction.ldr(.x2, .{ .register = .{ .rn = .x1, .offset = Instruction.LoadStoreOffset.imm_pre_index(1) } }),
            .expected = 0b11_111_0_00_01_0_000000001_11_00001_00010,
        },
        .{ // ldr x2, [x1], #-1
            .inst = Instruction.ldr(.x2, .{ .register = .{ .rn = .x1, .offset = Instruction.LoadStoreOffset.imm_post_index(-1) } }),
            .expected = 0b11_111_0_00_01_0_111111111_01_00001_00010,
        },
        .{ // ldr x2, [x1], (x3)
            .inst = Instruction.ldr(.x2, .{ .register = .{ .rn = .x1, .offset = Instruction.LoadStoreOffset.reg(.x3) } }),
            .expected = 0b11_111_0_00_01_1_00011_011_0_10_00001_00010,
        },
        .{ // ldr x2, label
            .inst = Instruction.ldr(.x2, .{ .literal = 0x1 }),
            .expected = 0b01_011_0_00_0000000000000000001_00010,
        },
        .{ // ldrh x7, [x4], #0xaa
            .inst = Instruction.ldrh(.x7, .x4, .{ .offset = Instruction.LoadStoreOffset.imm_post_index(0xaa) }),
            .expected = 0b01_111_0_00_01_0_010101010_01_00100_00111,
        },
        .{ // ldrb x9, [x15, #0xff]!
            .inst = Instruction.ldrb(.x9, .x15, .{ .offset = Instruction.LoadStoreOffset.imm_pre_index(0xff) }),
            .expected = 0b00_111_0_00_01_0_011111111_11_01111_01001,
        },
        .{ // str x2, [x1]
            .inst = Instruction.str(.x2, .x1, .{}),
            .expected = 0b11_111_0_01_00_000000000000_00001_00010,
        },
        .{ // str x2, [x1], (x3)
            .inst = Instruction.str(.x2, .x1, .{ .offset = Instruction.LoadStoreOffset.reg(.x3) }),
            .expected = 0b11_111_0_00_00_1_00011_011_0_10_00001_00010,
        },
        .{ // strh w0, [x1]
            .inst = Instruction.strh(.w0, .x1, .{}),
            .expected = 0b01_111_0_01_00_000000000000_00001_00000,
        },
        .{ // strb w8, [x9]
            .inst = Instruction.strb(.w8, .x9, .{}),
            .expected = 0b00_111_0_01_00_000000000000_01001_01000,
        },
        .{ // adr x2, #0x8
            .inst = Instruction.adr(.x2, 0x8),
            .expected = 0b0_00_10000_0000000000000000010_00010,
        },
        .{ // adr x2, -#0x8
            .inst = Instruction.adr(.x2, -0x8),
            .expected = 0b0_00_10000_1111111111111111110_00010,
        },
        .{ // adrp x2, #0x8
            .inst = Instruction.adrp(.x2, 0x8),
            .expected = 0b1_00_10000_0000000000000000010_00010,
        },
        .{ // adrp x2, -#0x8
            .inst = Instruction.adrp(.x2, -0x8),
            .expected = 0b1_00_10000_1111111111111111110_00010,
        },
        .{ // stp x1, x2, [sp, #8]
            .inst = Instruction.stp(.x1, .x2, Register.sp, Instruction.LoadStorePairOffset.signed(8)),
            .expected = 0b10_101_0_010_0_0000001_00010_11111_00001,
        },
        .{ // ldp x1, x2, [sp, #8]
            .inst = Instruction.ldp(.x1, .x2, Register.sp, Instruction.LoadStorePairOffset.signed(8)),
            .expected = 0b10_101_0_010_1_0000001_00010_11111_00001,
        },
        .{ // stp x1, x2, [sp, #-16]!
            .inst = Instruction.stp(.x1, .x2, Register.sp, Instruction.LoadStorePairOffset.pre_index(-16)),
            .expected = 0b10_101_0_011_0_1111110_00010_11111_00001,
        },
        .{ // ldp x1, x2, [sp], #16
            .inst = Instruction.ldp(.x1, .x2, Register.sp, Instruction.LoadStorePairOffset.post_index(16)),
            .expected = 0b10_101_0_001_1_0000010_00010_11111_00001,
        },
        .{ // and x0, x4, x2
            .inst = Instruction.@"and"(.x0, .x4, .x2, .{}),
            .expected = 0b1_00_01010_00_0_00010_000000_00100_00000,
        },
        .{ // and x0, x4, x2, lsl #0x8
            .inst = Instruction.@"and"(.x0, .x4, .x2, .{ .shift = .lsl, .amount = 0x8 }),
            .expected = 0b1_00_01010_00_0_00010_001000_00100_00000,
        },
        .{ // add x0, x10, #10
            .inst = Instruction.add(.x0, .x10, 10, false),
            .expected = 0b1_0_0_100010_0_0000_0000_1010_01010_00000,
        },
        .{ // subs x0, x5, #11, lsl #12
            .inst = Instruction.subs(.x0, .x5, 11, true),
            .expected = 0b1_1_1_100010_1_0000_0000_1011_00101_00000,
        },
        .{ // b.hi #-4
            .inst = Instruction.bCond(.hi, -4),
            .expected = 0b0101010_0_1111111111111111111_0_1000,
        },
        .{ // cbz x10, #40
            .inst = Instruction.cbz(.x10, 40),
            .expected = 0b1_011010_0_0000000000000001010_01010,
        },
    };

    for (testcases) |case| {
        const actual = case.inst.toU32();
        testing.expectEqual(case.expected, actual);
    }
}

const mem = std.mem;
const math = std.math;
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
                // TODO Finish function prologue and epilogue for aarch64.

                // stp fp, lr, [sp, #-16]!
                // mov fp, sp
                // sub sp, sp, #reloc
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.stp(
                    .x29,
                    .x30,
                    Register.sp,
                    Instruction.LoadStorePairOffset.pre_index(-16),
                ).toU32());
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.add(.x29, .xzr, 0, false).toU32());
                const backpatch_reloc = self.code.items.len;
                try self.code.resize(backpatch_reloc + 4);

                try CodegenUtils.dbgSetPrologueEnd(Self, self);

                try CodegenUtils.genBody(Self, self, self.mod_fn.body);

                // Backpatch stack offset
                const stack_end = self.max_end_stack;
                const aligned_stack_end = mem.alignForward(stack_end, self.stack_align);
                if (math.cast(u12, aligned_stack_end)) |size| {
                    writeInt(u32, self.code.items[backpatch_reloc..][0..4], Instruction.sub(.xzr, .xzr, size, false).toU32());
                } else |_| {
                    return CodegenUtils.failSymbol(Self, self, "TODO AArch64: allow larger stacks", .{});
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
                        if (math.cast(i28, amt)) |offset| {
                            writeInt(u32, self.code.items[jmp_reloc..][0..4], Instruction.b(offset).toU32());
                        } else |err| {
                            return CodegenUtils.failSymbol(Self, self, "exitlude jump is too large", .{});
                        }
                    }
                }

                // ldp fp, lr, [sp], #16
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.ldp(
                    .x29,
                    .x30,
                    Register.sp,
                    Instruction.LoadStorePairOffset.post_index(16),
                ).toU32());
                // add sp, sp, #stack_size
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.add(.xzr, .xzr, @intCast(u12, aligned_stack_end), false).toU32());
                // ret lr
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.ret(null).toU32());
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
                .stack_offset => |offset| {},
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
            switch (arch) {
                .aarch64 => {
                    mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.brk(1).toU32());
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement @breakpoint() for {}", .{self.target.cpu.arch}),
            }
            return .none;
        }

        fn genCall(self: *Self, inst: *ir.Inst.Call) !MCValue {
            var info = try self.resolveCallingConventionValues(inst.base.src, inst.func.ty);
            defer info.deinit(self);

            // Due to incremental compilation, how function calls are generated depends
            // on linking.
            if (self.bin_file.tag == link.File.Elf.base_tag or self.bin_file.tag == link.File.Coff.base_tag) {
                switch (arch) {
                    .aarch64 => {
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

                                try self.genSetReg(inst.base.src, Type.initTag(.usize), .x30, .{ .memory = got_addr });

                                writeInt(u32, try self.code.addManyAsArray(4), Instruction.blr(.x30).toU32());
                            } else if (func_value.castTag(.extern_fn)) |_| {
                                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling extern functions", .{});
                            } else {
                                return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling bitcasted functions", .{});
                            }
                        } else {
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling runtime known function pointer", .{});
                        }
                    },
                    else => return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement call for {}", .{self.target.cpu.arch}),
                }
            } else if (self.bin_file.cast(link.File.MachO)) |macho_file| {
                for (info.args) |mc_arg, arg_i| {
                    const arg = inst.args[arg_i];
                    const arg_mcv = try CodegenUtils.resolveInst(Self, self, inst.args[arg_i]);
                    // Here we do not use setRegOrMem even though the logic is similar, because
                    // the function call will move the stack pointer, so the offsets are different.
                    switch (mc_arg) {
                        .none => continue,
                        .register => |reg| {
                            try self.register_manager.getRegWithoutTracking(reg);
                            try self.genSetReg(arg.src, arg.ty, reg, arg_mcv);
                        },
                        .stack_offset => {
                            // Here we need to emit instructions like this:
                            // mov     qword ptr [rsp + stack_offset], x
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling with parameters in memory", .{});
                        },
                        .ptr_stack_offset => {
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling with MCValue.ptr_stack_offset arg", .{});
                        },
                        .ptr_embedded_in_code => {
                            return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling with MCValue.ptr_embedded_in_code arg", .{});
                        },
                        .undef => unreachable,
                        .immediate => unreachable,
                        .unreach => unreachable,
                        .dead => unreachable,
                        .embedded_in_code => unreachable,
                        .memory => unreachable,
                        .compare_flags_signed => unreachable,
                        .compare_flags_unsigned => unreachable,
                    }
                }

                if (inst.func.value()) |func_value| {
                    if (func_value.castTag(.function)) |func_payload| {
                        const func = func_payload.data;
                        const got_addr = blk: {
                            const seg = macho_file.load_commands.items[macho_file.data_const_segment_cmd_index.?].Segment;
                            const got = seg.sections.items[macho_file.got_section_index.?];
                            break :blk got.addr + func.owner_decl.link.macho.offset_table_index * @sizeOf(u64);
                        };
                        log.debug("got_addr = 0x{x}", .{got_addr});
                        switch (arch) {
                            .aarch64 => {
                                try self.genSetReg(inst.base.src, Type.initTag(.u64), .x30, .{ .memory = got_addr });
                                // blr x30
                                writeInt(u32, try self.code.addManyAsArray(4), Instruction.blr(.x30).toU32());
                            },
                            else => unreachable, // unsupported architecture on MachO
                        }
                    } else if (func_value.castTag(.extern_fn)) |func_payload| {
                        const decl = func_payload.data;
                        const decl_name = try std.fmt.allocPrint(self.bin_file.allocator, "_{s}", .{decl.name});
                        defer self.bin_file.allocator.free(decl_name);
                        const already_defined = macho_file.lazy_imports.contains(decl_name);
                        const symbol: u32 = if (macho_file.lazy_imports.getIndex(decl_name)) |index|
                            @intCast(u32, index)
                        else
                            try macho_file.addExternSymbol(decl_name);
                        const start = self.code.items.len;
                        const len: usize = blk: {
                            switch (arch) {
                                .aarch64 => {
                                    // bl
                                    writeInt(u32, try self.code.addManyAsArray(4), 0);
                                    break :blk 4;
                                },
                                else => unreachable, // unsupported architecture on MachO
                            }
                        };
                        try macho_file.stub_fixups.append(self.bin_file.allocator, .{
                            .symbol = symbol,
                            .already_defined = already_defined,
                            .start = start,
                            .len = len,
                        });
                        // We mark the space and fix it up later.
                    } else {
                        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling bitcasted functions", .{});
                    }
                } else {
                    return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement calling runtime known function pointer", .{});
                }
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
            switch (arch) {
                .aarch64 => {
                    // Just add space for an instruction, patch this later
                    try self.code.resize(self.code.items.len + 4);
                    try self.exitlude_jump_relocs.append(self.gpa, self.code.items.len - 4);
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement return for {}", .{self.target.cpu.arch}),
            }
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
            if (math.cast(i28, @intCast(i32, index) - @intCast(i32, self.code.items.len + 8))) |delta| {
                writeInt(u32, try self.code.addManyAsArray(4), Instruction.b(delta).toU32());
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
            // Emit a jump with a relocation. It will be patched up after the block ends.
            try block.codegen.relocs.ensureCapacity(self.gpa, block.codegen.relocs.items.len + 1);

            return CodegenUtils.fail(Self, self, src, "TODO implement brvoid for {}", .{self.target.cpu.arch});
        }

        fn genAsm(self: *Self, inst: *ir.Inst.Assembly) !MCValue {
            if (!inst.is_volatile and inst.base.isUnused())
                return MCValue.dead;
            switch (arch) {
                .aarch64 => {
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
                        mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.svc(0x0).toU32());
                    } else if (mem.eql(u8, inst.asm_source, "svc #0x80")) {
                        mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.svc(0x80).toU32());
                    } else {
                        return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement support for more aarch64 assembly instructions", .{});
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
                },
                else => return CodegenUtils.fail(Self, self, inst.base.src, "TODO implement inline asm support for more architectures", .{}),
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
                        1, 2, 4, 8 => {
                            const offset = if (math.cast(i9, adj_off)) |imm|
                                Instruction.LoadStoreOffset.imm_post_index(-imm)
                            else |_|
                                Instruction.LoadStoreOffset.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u64), MCValue{ .immediate = adj_off }));
                            const rn: Register = switch (arch) {
                                .aarch64, .aarch64_be => .x29,
                                .aarch64_32 => .w29,
                                else => unreachable,
                            };
                            const str = switch (abi_size) {
                                1 => Instruction.strb,
                                2 => Instruction.strh,
                                4, 8 => Instruction.str,
                                else => unreachable, // unexpected abi size
                            };

                            writeInt(u32, try self.code.addManyAsArray(4), str(reg, rn, .{
                                .offset = offset,
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
            switch (arch) {
                .aarch64 => switch (mcv) {
                    .dead => unreachable,
                    .ptr_stack_offset => unreachable,
                    .ptr_embedded_in_code => unreachable,
                    .unreach, .none => return, // Nothing to do.
                    .undef => {
                        if (!self.wantSafety())
                            return; // The already existing value will do just fine.
                        // Write the debug undefined value.
                        switch (reg.size()) {
                            32 => return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaaaaaa }),
                            64 => return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaaaaaaaaaaaaaa }),
                            else => unreachable, // unexpected register size
                        }
                    },
                    .immediate => |x| {
                        if (x <= math.maxInt(u16)) {
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movz(reg, @intCast(u16, x), 0).toU32());
                        } else if (x <= math.maxInt(u32)) {
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movz(reg, @truncate(u16, x), 0).toU32());
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movk(reg, @intCast(u16, x >> 16), 16).toU32());
                        } else if (x <= math.maxInt(u32)) {
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movz(reg, @truncate(u16, x), 0).toU32());
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movk(reg, @truncate(u16, x >> 16), 16).toU32());
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movk(reg, @intCast(u16, x >> 32), 32).toU32());
                        } else {
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movz(reg, @truncate(u16, x), 0).toU32());
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movk(reg, @truncate(u16, x >> 16), 16).toU32());
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movk(reg, @truncate(u16, x >> 32), 32).toU32());
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.movk(reg, @intCast(u16, x >> 48), 48).toU32());
                        }
                    },
                    .register => |src_reg| {
                        // If the registers are the same, nothing to do.
                        if (src_reg.id() == reg.id())
                            return;

                        // mov reg, src_reg
                        writeInt(u32, try self.code.addManyAsArray(4), Instruction.orr(
                            reg,
                            .xzr,
                            src_reg,
                            Instruction.Shift.none,
                        ).toU32());
                    },
                    .memory => |addr| {
                        if (self.bin_file.options.pie) {
                            // PC-relative displacement to the entry in the GOT table.
                            // TODO we should come up with our own, backend independent relocation types
                            // which each backend (Elf, MachO, etc.) would then translate into an actual
                            // fixup when linking.
                            // adrp reg, pages
                            if (self.bin_file.cast(link.File.MachO)) |macho_file| {
                                try macho_file.pie_fixups.append(self.bin_file.allocator, .{
                                    .target_addr = addr,
                                    .offset = self.code.items.len,
                                    .size = 4,
                                });
                            } else {
                                return CodegenUtils.fail(Self, self, src, "TODO implement genSetReg for PIE GOT indirection on this platform", .{});
                            }
                            mem.writeIntLittle(
                                u32,
                                try self.code.addManyAsArray(4),
                                Instruction.adrp(reg, 0).toU32(),
                            );
                            // ldr reg, reg, offset
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.ldr(reg, .{
                                .register = .{
                                    .rn = reg,
                                    .offset = Instruction.LoadStoreOffset.imm(0),
                                },
                            }).toU32());
                        } else {
                            // The value is in memory at a hard-coded address.
                            // If the type is a pointer, it means the pointer address is at this memory location.
                            try self.genSetReg(src, Type.initTag(.usize), reg, .{ .immediate = addr });
                            mem.writeIntLittle(u32, try self.code.addManyAsArray(4), Instruction.ldr(reg, .{ .register = .{ .rn = reg } }).toU32());
                        }
                    },
                    .stack_offset => |unadjusted_off| {
                        // TODO: maybe addressing from sp instead of fp
                        const abi_size = ty.abiSize(self.target.*);
                        const adj_off = unadjusted_off + abi_size;

                        const rn: Register = switch (arch) {
                            .aarch64, .aarch64_be => .x29,
                            .aarch64_32 => .w29,
                            else => unreachable,
                        };

                        const offset = if (math.cast(i9, adj_off)) |imm|
                            Instruction.LoadStoreOffset.imm_post_index(-imm)
                        else |_|
                            Instruction.LoadStoreOffset.reg(try CodegenUtils.copyToTmpRegister(Self, self, src, Type.initTag(.u64), MCValue{ .immediate = adj_off }));

                        switch (abi_size) {
                            1, 2 => {
                                const ldr = switch (abi_size) {
                                    1 => Instruction.ldrb,
                                    2 => Instruction.ldrh,
                                    else => unreachable, // unexpected abi size
                                };

                                writeInt(u32, try self.code.addManyAsArray(4), ldr(reg, rn, .{
                                    .offset = offset,
                                }).toU32());
                            },
                            4, 8 => {
                                writeInt(u32, try self.code.addManyAsArray(4), Instruction.ldr(reg, .{ .register = .{
                                    .rn = rn,
                                    .offset = offset,
                                } }).toU32());
                            },
                            else => return CodegenUtils.fail(Self, self, src, "TODO implement genSetReg other types abi_size={}", .{abi_size}),
                        }
                    },
                    else => return CodegenUtils.fail(Self, self, src, "TODO implement genSetReg for aarch64 {}", .{mcv}),
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement getSetReg for {}", .{self.target.cpu.arch}),
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

            switch (cc) {
                .Naked => {
                    assert(result.args.len == 0);
                    result.return_value = .{ .unreach = {} };
                    result.stack_byte_count = 0;
                    result.stack_align = 1;
                    return result;
                },
                .Unspecified, .C => {
                    // ARM64 Procedure Call Standard
                    var ncrn: usize = 0; // Next Core Register Number
                    var nsaa: u32 = 0; // Next stacked argument address

                    for (param_types) |ty, i| {
                        // We round up NCRN only for non-Apple platforms which allow the 16-byte aligned
                        // values to spread across odd-numbered registers.
                        if (ty.abiAlignment(self.target.*) == 16 and !self.target.isDarwin()) {
                            // Round up NCRN to the next even number
                            ncrn += ncrn % 2;
                        }

                        const param_size = @intCast(u32, ty.abiSize(self.target.*));
                        if (std.math.divCeil(u32, param_size, 8) catch unreachable <= 8 - ncrn) {
                            if (param_size <= 8) {
                                result.args[i] = .{ .register = c_abi_int_param_regs[ncrn] };
                                ncrn += 1;
                            } else {
                                return CodegenUtils.fail(Self, self, src, "TODO MCValues with multiple registers", .{});
                            }
                        } else if (ncrn < 8 and nsaa == 0) {
                            return CodegenUtils.fail(Self, self, src, "TODO MCValues split between registers and stack", .{});
                        } else {
                            ncrn = 8;
                            // TODO Apple allows the arguments on the stack to be non-8-byte aligned provided
                            // that the entire stack space consumed by the arguments is 8-byte aligned.
                            if (ty.abiAlignment(self.target.*) == 8) {
                                if (nsaa % 8 != 0) {
                                    nsaa += 8 - (nsaa % 8);
                                }
                            }

                            result.args[i] = .{ .stack_offset = nsaa };
                            nsaa += param_size;
                        }
                    }

                    result.stack_byte_count = nsaa;
                    result.stack_align = 16;
                },
                else => return CodegenUtils.fail(Self, self, src, "TODO implement function parameters for {} on aarch64", .{cc}),
            }

            if (ret_ty.zigTypeTag() == .NoReturn) {
                result.return_value = .{ .unreach = {} };
            } else if (!ret_ty.hasCodeGenBits()) {
                result.return_value = .{ .none = {} };
            } else switch (cc) {
                .Naked => unreachable,
                .Unspecified, .C => {
                    const ret_ty_size = @intCast(u32, ret_ty.abiSize(self.target.*));
                    if (ret_ty_size <= 8) {
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
