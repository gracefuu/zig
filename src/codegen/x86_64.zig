const std = @import("std");
const testing = std.testing;
const mem = std.mem;
const assert = std.debug.assert;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Type = @import("../type.zig").Type;
const DW = std.dwarf;

// zig fmt: off

/// Definitions of all of the x64 registers. The order is semantically meaningful.
/// The registers are defined such that IDs go in descending order of 64-bit,
/// 32-bit, 16-bit, and then 8-bit, and each set contains exactly sixteen
/// registers. This results in some useful properties:
///
/// Any 64-bit register can be turned into its 32-bit form by adding 16, and
/// vice versa. This also works between 32-bit and 16-bit forms. With 8-bit, it
/// works for all except for sp, bp, si, and di, which do *not* have an 8-bit
/// form.
///
/// If (register & 8) is set, the register is extended.
///
/// The ID can be easily determined by figuring out what range the register is
/// in, and then subtracting the base.
pub const Register = enum(u8) {
    // 0 through 15, 64-bit registers. 8-15 are extended.
    // id is just the int value.
    rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
    r8, r9, r10, r11, r12, r13, r14, r15,

    // 16 through 31, 32-bit registers. 24-31 are extended.
    // id is int value - 16.
    eax, ecx, edx, ebx, esp, ebp, esi, edi, 
    r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d,

    // 32-47, 16-bit registers. 40-47 are extended.
    // id is int value - 32.
    ax, cx, dx, bx, sp, bp, si, di,
    r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w,
    
    // 48-63, 8-bit registers. 56-63 are extended.
    // id is int value - 48.
    al, cl, dl, bl, ah, ch, dh, bh,
    r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b,

    /// Returns the bit-width of the register.
    pub fn size(self: Register) u7 {
        return switch (@enumToInt(self)) {
            0...15 => 64,
            16...31 => 32,
            32...47 => 16,
            48...64 => 8,
            else => unreachable,
        };
    }

    /// Returns whether the register is *extended*. Extended registers are the
    /// new registers added with amd64, r8 through r15. This also includes any
    /// other variant of access to those registers, such as r8b, r15d, and so
    /// on. This is needed because access to these registers requires special
    /// handling via the REX prefix, via the B or R bits, depending on context.
    pub fn isExtended(self: Register) bool {
        return @enumToInt(self) & 0x08 != 0;
    }

    /// This returns the 4-bit register ID, which is used in practically every
    /// opcode. Note that bit 3 (the highest bit) is *never* used directly in
    /// an instruction (@see isExtended), and requires special handling. The
    /// lower three bits are often embedded directly in instructions (such as
    /// the B8 variant of moves), or used in R/M bytes.
    pub fn id(self: Register) u4 {
        return @truncate(u4, @enumToInt(self));
    }

    /// Like id, but only returns the lower 3 bits.
    pub fn low_id(self: Register) u3 {
        return @truncate(u3, @enumToInt(self));
    }

    /// Returns the index into `callee_preserved_regs`.
    pub fn allocIndex(self: Register) ?u4 {
        return switch (self) {
            .rax, .eax, .ax, .al => 0,
            .rcx, .ecx, .cx, .cl => 1,
            .rdx, .edx, .dx, .dl => 2,
            .rsi, .esi, .si  => 3,
            .rdi, .edi, .di => 4,
            .r8, .r8d, .r8w, .r8b => 5,
            .r9, .r9d, .r9w, .r9b => 6,
            .r10, .r10d, .r10w, .r10b => 7,
            .r11, .r11d, .r11w, .r11b => 8,
            else => null,
        };
    }

    /// Convert from any register to its 64 bit alias.
    pub fn to64(self: Register) Register {
        return @intToEnum(Register, self.id());
    }

    /// Convert from any register to its 32 bit alias.
    pub fn to32(self: Register) Register {
        return @intToEnum(Register, @as(u8, self.id()) + 16);
    }

    /// Convert from any register to its 16 bit alias.
    pub fn to16(self: Register) Register {
        return @intToEnum(Register, @as(u8, self.id()) + 32);
    }

    /// Convert from any register to its 8 bit alias.
    pub fn to8(self: Register) Register {
        return @intToEnum(Register, @as(u8, self.id()) + 48);
    }

    pub fn dwarfLocOp(self: Register) u8 {
        return switch (self.to64()) {
            .rax => DW.OP_reg0,
            .rdx => DW.OP_reg1,
            .rcx => DW.OP_reg2,
            .rbx => DW.OP_reg3,
            .rsi => DW.OP_reg4,
            .rdi => DW.OP_reg5,
            .rbp => DW.OP_reg6,
            .rsp => DW.OP_reg7,

            .r8 => DW.OP_reg8,
            .r9 => DW.OP_reg9,
            .r10 => DW.OP_reg10,
            .r11 => DW.OP_reg11,
            .r12 => DW.OP_reg12,
            .r13 => DW.OP_reg13,
            .r14 => DW.OP_reg14,
            .r15 => DW.OP_reg15,

            else => unreachable,
        };
    }
};

// zig fmt: on

/// These registers belong to the called function.
pub const callee_preserved_regs = [_]Register{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9, .r10, .r11 };
pub const c_abi_int_param_regs = [_]Register{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
pub const c_abi_int_return_regs = [_]Register{ .rax, .rdx };

/// Encoding helper functions for x86_64 instructions
///
/// Many of these helpers do very little, but they can help make things
/// slightly more readable with more descriptive field names / function names.
///
/// Some of them also have asserts to ensure that we aren't doing dumb things.
/// For example, trying to use register 4 (esp) in an indirect modr/m byte is illegal,
/// you need to encode it with an SIB byte.
///
/// Note that ALL of these helper functions will assume capacity,
/// so ensure that the `code` has sufficient capacity before using them.
/// The `init` method is the recommended way to ensure capacity.
pub const Encoder = struct {
    /// Non-owning reference to the code array
    code: *ArrayList(u8),

    const Self = @This();

    /// Wrap `code` in Encoder to make it easier to call these helper functions
    ///
    /// maximum_inst_size should contain the maximum number of bytes
    /// that the encoded instruction will take.
    /// This is because the helper functions will assume capacity
    /// in order to avoid bounds checking.
    pub fn init(code: *ArrayList(u8), maximum_inst_size: u8) !Self {
        try code.ensureCapacity(code.items.len + maximum_inst_size);
        return Self{ .code = code };
    }

    /// Directly write a number to the code array with big endianness
    pub fn writeIntBig(self: Self, comptime T: type, value: T) void {
        mem.writeIntBig(
            T,
            self.code.addManyAsArrayAssumeCapacity(@divExact(@typeInfo(T).Int.bits, 8)),
            value,
        );
    }

    /// Directly write a number to the code array with little endianness
    pub fn writeIntLittle(self: Self, comptime T: type, value: T) void {
        mem.writeIntLittle(
            T,
            self.code.addManyAsArrayAssumeCapacity(@divExact(@typeInfo(T).Int.bits, 8)),
            value,
        );
    }

    // --------
    // Prefixes
    // --------

    pub const LegacyPrefixes = packed struct {
        /// LOCK
        prefix_f0: bool = false,
        /// REPNZ, REPNE, REP, Scalar Double-precision
        prefix_f2: bool = false,
        /// REPZ, REPE, REP, Scalar Single-precision
        prefix_f3: bool = false,

        /// CS segment override or Branch not taken
        prefix_2e: bool = false,
        /// DS segment override
        prefix_36: bool = false,
        /// ES segment override
        prefix_26: bool = false,
        /// FS segment override
        prefix_64: bool = false,
        /// GS segment override
        prefix_65: bool = false,

        /// Branch taken
        prefix_3e: bool = false,

        /// Operand size override (enables 16 bit operation)
        prefix_66: bool = false,

        /// Address size override (enables 16 bit address size)
        prefix_67: bool = false,

        padding: u5 = 0,
    };

    /// Encodes legacy prefixes
    pub fn legacyPrefixes(self: Self, prefixes: LegacyPrefixes) void {
        if (@bitCast(u16, prefixes) != 0) {
            // Hopefully this path isn't taken very often, so we'll do it the slow way for now

            // LOCK
            if (prefixes.prefix_f0) self.code.appendAssumeCapacity(0xf0);
            // REPNZ, REPNE, REP, Scalar Double-precision
            if (prefixes.prefix_f2) self.code.appendAssumeCapacity(0xf2);
            // REPZ, REPE, REP, Scalar Single-precision
            if (prefixes.prefix_f3) self.code.appendAssumeCapacity(0xf3);

            // CS segment override or Branch not taken
            if (prefixes.prefix_2e) self.code.appendAssumeCapacity(0x2e);
            // DS segment override
            if (prefixes.prefix_36) self.code.appendAssumeCapacity(0x36);
            // ES segment override
            if (prefixes.prefix_26) self.code.appendAssumeCapacity(0x26);
            // FS segment override
            if (prefixes.prefix_64) self.code.appendAssumeCapacity(0x64);
            // GS segment override
            if (prefixes.prefix_65) self.code.appendAssumeCapacity(0x65);

            // Branch taken
            if (prefixes.prefix_3e) self.code.appendAssumeCapacity(0x3e);

            // Operand size override
            if (prefixes.prefix_66) self.code.appendAssumeCapacity(0x66);

            // Address size override
            if (prefixes.prefix_67) self.code.appendAssumeCapacity(0x67);
        }
    }

    /// Use 16 bit operand size
    ///
    /// Note that this flag is overridden by REX.W, if both are present.
    pub fn prefix16BitMode(self: Self) void {
        self.code.appendAssumeCapacity(0x66);
    }

    /// From section 2.2.1.2 of the manual, REX is encoded as b0100WRXB
    pub const Rex = struct {
        /// Wide, enables 64-bit operation
        w: bool = false,
        /// Extends the reg field in the ModR/M byte
        r: bool = false,
        /// Extends the index field in the SIB byte
        x: bool = false,
        /// Extends the r/m field in the ModR/M byte,
        ///      or the base field in the SIB byte,
        ///      or the reg field in the Opcode byte
        b: bool = false,
    };

    /// Encodes a REX prefix byte given all the fields
    ///
    /// Use this byte whenever you need 64 bit operation,
    /// or one of reg, index, r/m, base, or opcode-reg might be extended.
    ///
    /// See struct `Rex` for a description of each field.
    ///
    /// Does not add a prefix byte if none of the fields are set!
    pub fn rex(self: Self, byte: Rex) void {
        var value: u8 = 0b0100_0000;

        if (byte.w) value |= 0b1000;
        if (byte.r) value |= 0b0100;
        if (byte.x) value |= 0b0010;
        if (byte.b) value |= 0b0001;

        if (value != 0b0100_0000) {
            self.code.appendAssumeCapacity(value);
        }
    }

    // ------
    // Opcode
    // ------

    /// Encodes a 1 byte opcode
    pub fn opcode_1byte(self: Self, opcode: u8) void {
        self.code.appendAssumeCapacity(opcode);
    }

    /// Encodes a 2 byte opcode
    ///
    /// e.g. IMUL has the opcode 0x0f 0xaf, so you use
    ///
    /// encoder.opcode_2byte(0x0f, 0xaf);
    pub fn opcode_2byte(self: Self, prefix: u8, opcode: u8) void {
        self.code.appendAssumeCapacity(prefix);
        self.code.appendAssumeCapacity(opcode);
    }

    /// Encodes a 1 byte opcode with a reg field
    ///
    /// Remember to add a REX prefix byte if reg is extended!
    pub fn opcode_withReg(self: Self, opcode: u8, reg: u3) void {
        assert(opcode & 0b111 == 0);
        self.code.appendAssumeCapacity(opcode | reg);
    }

    // ------
    // ModR/M
    // ------

    /// Construct a ModR/M byte given all the fields
    ///
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm(self: Self, mod: u2, reg_or_opx: u3, rm: u3) void {
        self.code.appendAssumeCapacity(
            @as(u8, mod) << 6 | @as(u8, reg_or_opx) << 3 | rm,
        );
    }

    /// Construct a ModR/M byte using direct r/m addressing
    /// r/m effective address: r/m
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_direct(self: Self, reg_or_opx: u3, rm: u3) void {
        self.modRm(0b11, reg_or_opx, rm);
    }

    /// Construct a ModR/M byte using indirect r/m addressing
    /// r/m effective address: [r/m]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_indirectDisp0(self: Self, reg_or_opx: u3, rm: u3) void {
        assert(rm != 4 and rm != 5);
        self.modRm(0b00, reg_or_opx, rm);
    }

    /// Construct a ModR/M byte using indirect SIB addressing
    /// r/m effective address: [SIB]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_SIBDisp0(self: Self, reg_or_opx: u3) void {
        self.modRm(0b00, reg_or_opx, 0b100);
    }

    /// Construct a ModR/M byte using RIP-relative addressing
    /// r/m effective address: [RIP + disp32]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_RIPDisp32(self: Self, reg_or_opx: u3) void {
        self.modRm(0b00, reg_or_opx, 0b101);
    }

    /// Construct a ModR/M byte using indirect r/m with a 8bit displacement
    /// r/m effective address: [r/m + disp8]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_indirectDisp8(self: Self, reg_or_opx: u3, rm: u3) void {
        assert(rm != 4);
        self.modRm(0b01, reg_or_opx, rm);
    }

    /// Construct a ModR/M byte using indirect SIB with a 8bit displacement
    /// r/m effective address: [SIB + disp8]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_SIBDisp8(self: Self, reg_or_opx: u3) void {
        self.modRm(0b01, reg_or_opx, 0b100);
    }

    /// Construct a ModR/M byte using indirect r/m with a 32bit displacement
    /// r/m effective address: [r/m + disp32]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_indirectDisp32(self: Self, reg_or_opx: u3, rm: u3) void {
        assert(rm != 4);
        self.modRm(0b10, reg_or_opx, rm);
    }

    /// Construct a ModR/M byte using indirect SIB with a 32bit displacement
    /// r/m effective address: [SIB + disp32]
    ///
    /// Note reg's effective address is always just reg for the ModR/M byte.
    /// Remember to add a REX prefix byte if reg or rm are extended!
    pub fn modRm_SIBDisp32(self: Self, reg_or_opx: u3) void {
        self.modRm(0b10, reg_or_opx, 0b100);
    }

    // ---
    // SIB
    // ---

    /// Construct a SIB byte given all the fields
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib(self: Self, scale: u2, index: u3, base: u3) void {
        self.code.appendAssumeCapacity(
            @as(u8, scale) << 6 | @as(u8, index) << 3 | base,
        );
    }

    /// Construct a SIB byte with scale * index + base, no frills.
    /// r/m effective address: [base + scale * index]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_scaleIndexBase(self: Self, scale: u2, index: u3, base: u3) void {
        assert(base != 5);

        self.sib(scale, index, base);
    }

    /// Construct a SIB byte with scale * index + disp32
    /// r/m effective address: [scale * index + disp32]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_scaleIndexDisp32(self: Self, scale: u2, index: u3) void {
        assert(index != 4);

        // scale is actually ignored
        // index = 4 means no index
        // base = 5 means no base, if mod == 0.
        self.sib(scale, index, 5);
    }

    /// Construct a SIB byte with just base
    /// r/m effective address: [base]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_base(self: Self, base: u3) void {
        assert(base != 5);

        // scale is actually ignored
        // index = 4 means no index
        self.sib(0, 4, base);
    }

    /// Construct a SIB byte with just disp32
    /// r/m effective address: [disp32]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_disp32(self: Self) void {
        // scale is actually ignored
        // index = 4 means no index
        // base = 5 means no base, if mod == 0.
        self.sib(0, 4, 5);
    }

    /// Construct a SIB byte with scale * index + base + disp8
    /// r/m effective address: [base + scale * index + disp8]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_scaleIndexBaseDisp8(self: Self, scale: u2, index: u3, base: u3) void {
        self.sib(scale, index, base);
    }

    /// Construct a SIB byte with base + disp8, no index
    /// r/m effective address: [base + disp8]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_baseDisp8(self: Self, base: u3) void {
        // scale is ignored
        // index = 4 means no index
        self.sib(0, 4, base);
    }

    /// Construct a SIB byte with scale * index + base + disp32
    /// r/m effective address: [base + scale * index + disp32]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_scaleIndexBaseDisp32(self: Self, scale: u2, index: u3, base: u3) void {
        self.sib(scale, index, base);
    }

    /// Construct a SIB byte with base + disp32, no index
    /// r/m effective address: [base + disp32]
    ///
    /// Remember to add a REX prefix byte if index or base are extended!
    pub fn sib_baseDisp32(self: Self, base: u3) void {
        // scale is ignored
        // index = 4 means no index
        self.sib(0, 4, base);
    }

    // -------------------------
    // Trivial (no bit fiddling)
    // -------------------------

    /// Encode an 8 bit immediate
    ///
    /// It is sign-extended to 64 bits by the cpu.
    pub fn imm8(self: Self, imm: i8) void {
        self.code.appendAssumeCapacity(@bitCast(u8, imm));
    }

    /// Encode an 8 bit displacement
    ///
    /// It is sign-extended to 64 bits by the cpu.
    pub fn disp8(self: Self, disp: i8) void {
        self.code.appendAssumeCapacity(@bitCast(u8, disp));
    }

    /// Encode an 16 bit immediate
    ///
    /// It is sign-extended to 64 bits by the cpu.
    pub fn imm16(self: Self, imm: i16) void {
        self.writeIntLittle(i16, imm);
    }

    /// Encode an 32 bit immediate
    ///
    /// It is sign-extended to 64 bits by the cpu.
    pub fn imm32(self: Self, imm: i32) void {
        self.writeIntLittle(i32, imm);
    }

    /// Encode an 32 bit displacement
    ///
    /// It is sign-extended to 64 bits by the cpu.
    pub fn disp32(self: Self, disp: i32) void {
        self.writeIntLittle(i32, disp);
    }

    /// Encode an 64 bit immediate
    ///
    /// It is sign-extended to 64 bits by the cpu.
    pub fn imm64(self: Self, imm: u64) void {
        self.writeIntLittle(u64, imm);
    }
};

test "x86_64 Encoder helpers" {
    var code = ArrayList(u8).init(testing.allocator);
    defer code.deinit();

    // simple integer multiplication

    // imul eax,edi
    // 0faf   c7
    {
        try code.resize(0);
        const encoder = try Encoder.init(&code, 4);
        encoder.rex(.{
            .r = Register.eax.isExtended(),
            .b = Register.edi.isExtended(),
        });
        encoder.opcode_2byte(0x0f, 0xaf);
        encoder.modRm_direct(
            Register.eax.low_id(),
            Register.edi.low_id(),
        );

        testing.expectEqualSlices(u8, &[_]u8{ 0x0f, 0xaf, 0xc7 }, code.items);
    }

    // simple mov

    // mov eax,edi
    // 89    f8
    {
        try code.resize(0);
        const encoder = try Encoder.init(&code, 3);
        encoder.rex(.{
            .r = Register.edi.isExtended(),
            .b = Register.eax.isExtended(),
        });
        encoder.opcode_1byte(0x89);
        encoder.modRm_direct(
            Register.edi.low_id(),
            Register.eax.low_id(),
        );

        testing.expectEqualSlices(u8, &[_]u8{ 0x89, 0xf8 }, code.items);
    }

    // signed integer addition of 32-bit sign extended immediate to 64 bit register

    // add rcx, 2147483647
    //
    // Using the following opcode: REX.W + 81 /0 id, we expect the following encoding
    //
    // 48       :  REX.W set for 64 bit operand (*r*cx)
    // 81       :  opcode for "<arithmetic> with immediate"
    // c1       :  id = rcx,
    //          :  c1 = 11  <-- mod = 11 indicates r/m is register (rcx)
    //          :       000 <-- opcode_extension = 0 because opcode extension is /0. /0 specifies ADD
    //          :       001 <-- 001 is rcx
    // ffffff7f :  2147483647
    {
        try code.resize(0);
        const encoder = try Encoder.init(&code, 7);
        encoder.rex(.{ .w = true }); // use 64 bit operation
        encoder.opcode_1byte(0x81);
        encoder.modRm_direct(
            0,
            Register.rcx.low_id(),
        );
        encoder.imm32(2147483647);

        testing.expectEqualSlices(u8, &[_]u8{ 0x48, 0x81, 0xc1, 0xff, 0xff, 0xff, 0x7f }, code.items);
    }
}

// TODO add these registers to the enum and populate dwarfLocOp
//    // Return Address register. This is stored in `0(%rsp, "")` and is not a physical register.
//    RA = (16, "RA"),
//
//    XMM0 = (17, "xmm0"),
//    XMM1 = (18, "xmm1"),
//    XMM2 = (19, "xmm2"),
//    XMM3 = (20, "xmm3"),
//    XMM4 = (21, "xmm4"),
//    XMM5 = (22, "xmm5"),
//    XMM6 = (23, "xmm6"),
//    XMM7 = (24, "xmm7"),
//
//    XMM8 = (25, "xmm8"),
//    XMM9 = (26, "xmm9"),
//    XMM10 = (27, "xmm10"),
//    XMM11 = (28, "xmm11"),
//    XMM12 = (29, "xmm12"),
//    XMM13 = (30, "xmm13"),
//    XMM14 = (31, "xmm14"),
//    XMM15 = (32, "xmm15"),
//
//    ST0 = (33, "st0"),
//    ST1 = (34, "st1"),
//    ST2 = (35, "st2"),
//    ST3 = (36, "st3"),
//    ST4 = (37, "st4"),
//    ST5 = (38, "st5"),
//    ST6 = (39, "st6"),
//    ST7 = (40, "st7"),
//
//    MM0 = (41, "mm0"),
//    MM1 = (42, "mm1"),
//    MM2 = (43, "mm2"),
//    MM3 = (44, "mm3"),
//    MM4 = (45, "mm4"),
//    MM5 = (46, "mm5"),
//    MM6 = (47, "mm6"),
//    MM7 = (48, "mm7"),
//
//    RFLAGS = (49, "rFLAGS"),
//    ES = (50, "es"),
//    CS = (51, "cs"),
//    SS = (52, "ss"),
//    DS = (53, "ds"),
//    FS = (54, "fs"),
//    GS = (55, "gs"),
//
//    FS_BASE = (58, "fs.base"),
//    GS_BASE = (59, "gs.base"),
//
//    TR = (62, "tr"),
//    LDTR = (63, "ldtr"),
//    MXCSR = (64, "mxcsr"),
//    FCW = (65, "fcw"),
//    FSW = (66, "fsw"),
//
//    XMM16 = (67, "xmm16"),
//    XMM17 = (68, "xmm17"),
//    XMM18 = (69, "xmm18"),
//    XMM19 = (70, "xmm19"),
//    XMM20 = (71, "xmm20"),
//    XMM21 = (72, "xmm21"),
//    XMM22 = (73, "xmm22"),
//    XMM23 = (74, "xmm23"),
//    XMM24 = (75, "xmm24"),
//    XMM25 = (76, "xmm25"),
//    XMM26 = (77, "xmm26"),
//    XMM27 = (78, "xmm27"),
//    XMM28 = (79, "xmm28"),
//    XMM29 = (80, "xmm29"),
//    XMM30 = (81, "xmm30"),
//    XMM31 = (82, "xmm31"),
//
//    K0 = (118, "k0"),
//    K1 = (119, "k1"),
//    K2 = (120, "k2"),
//    K3 = (121, "k3"),
//    K4 = (122, "k4"),
//    K5 = (123, "k5"),
//    K6 = (124, "k6"),
//    K7 = (125, "k7"),

const math = std.math;
const ir = @import("../ir.zig");
const Value = @import("../value.zig").Value;
const TypedValue = @import("../TypedValue.zig");
const link = @import("../link.zig");
const Module = @import("../Module.zig");
const Compilation = @import("../Compilation.zig");
const ErrorMsg = Module.ErrorMsg;
const Target = std.Target;
const trace = @import("../tracy.zig").trace;
const leb128 = std.leb;
const log = std.log.scoped(.codegen);
const build_options = @import("build_options");
const LazySrcLoc = Module.LazySrcLoc;
const RegisterManager = @import("../register_manager.zig").RegisterManager;

const X8664Encoder = Encoder;

const Codegen = @import("../codegen.zig");
const BlockData = Codegen.BlockData;
const AnyMCValue = Codegen.AnyMCValue;
const Reloc = Codegen.Reloc;
const Result = Codegen.Result;
const GenerateSymbolError = Codegen.GenerateSymbolError;
const DebugInfoOutput = Codegen.DebugInfoOutput;

const InnerError = error{
    OutOfMemory,
    CodegenFail,
};

const arch = std.Target.Cpu.Arch.x86_64;

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
        try self.code.ensureCapacity(self.code.items.len + 11);

        const cc = self.fn_type.fnCallingConvention();
        if (cc != .Naked) {
            // We want to subtract the aligned stack frame size from rsp here, but we don't
            // yet know how big it will be, so we leave room for a 4-byte stack size.
            // TODO During semantic analysis, check if there are no function calls. If there
            // are none, here we can omit the part where we subtract and then add rsp.
            self.code.appendSliceAssumeCapacity(&[_]u8{
                0x55, // push rbp
                0x48, 0x89, 0xe5, // mov rbp, rsp
                0x48, 0x81, 0xec, // sub rsp, imm32 (with reloc)
            });
            const reloc_index = self.code.items.len;
            self.code.items.len += 4;

            try self.dbgSetPrologueEnd();
            try self.genBody(self.mod_fn.body);

            const stack_end = self.max_end_stack;
            if (stack_end > math.maxInt(i32))
                return self.failSymbol("too much stack used in call parameters", .{});
            const aligned_stack_end = mem.alignForward(stack_end, self.stack_align);
            mem.writeIntLittle(u32, self.code.items[reloc_index..][0..4], @intCast(u32, aligned_stack_end));

            if (self.code.items.len >= math.maxInt(i32)) {
                return self.failSymbol("unable to perform relocation: jump too far", .{});
            }
            if (self.exitlude_jump_relocs.items.len == 1) {
                self.code.items.len -= 5;
            } else for (self.exitlude_jump_relocs.items) |jmp_reloc| {
                const amt = self.code.items.len - (jmp_reloc + 4);
                const s32_amt = @intCast(i32, amt);
                mem.writeIntLittle(i32, self.code.items[jmp_reloc..][0..4], s32_amt);
            }

            // Important to be after the possible self.code.items.len -= 5 above.
            try self.dbgSetEpilogueBegin();

            try self.code.ensureCapacity(self.code.items.len + 9);
            // add rsp, x
            if (aligned_stack_end > math.maxInt(i8)) {
                // example: 48 81 c4 ff ff ff 7f  add    rsp,0x7fffffff
                self.code.appendSliceAssumeCapacity(&[_]u8{ 0x48, 0x81, 0xc4 });
                const x = @intCast(u32, aligned_stack_end);
                mem.writeIntLittle(u32, self.code.addManyAsArrayAssumeCapacity(4), x);
            } else if (aligned_stack_end != 0) {
                // example: 48 83 c4 7f           add    rsp,0x7f
                const x = @intCast(u8, aligned_stack_end);
                self.code.appendSliceAssumeCapacity(&[_]u8{ 0x48, 0x83, 0xc4, x });
            }

            self.code.appendSliceAssumeCapacity(&[_]u8{
                0x5d, // pop rbp
                0xc3, // ret
            });
        } else {
            try self.dbgSetPrologueEnd();
            try self.genBody(self.mod_fn.body);
            try self.dbgSetEpilogueBegin();
        }

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
        return self.fail(inst.base.src, "TODO implement floatCast for {}", .{arch});
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

        return self.fail(inst.base.src, "TODO implement intCast for {}", .{arch});
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

        var imm = ir.Inst.Constant{
            .base = .{
                .tag = .constant,
                .deaths = 0,
                .ty = inst.operand.ty,
                .src = inst.operand.src,
            },
            .val = Value.initTag(.bool_true),
        };
        return try self.genX8664BinMath(&inst.base, inst.operand, &imm.base);
    }

    fn genAdd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return try self.genX8664BinMath(&inst.base, inst.lhs, inst.rhs);
    }

    fn genAddWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement addwrap for {}", .{arch});
    }

    fn genMul(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return try self.genX8664BinMath(&inst.base, inst.lhs, inst.rhs);
    }

    fn genMulWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement mulwrap for {}", .{arch});
    }

    fn genDiv(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement div for {}", .{arch});
    }

    fn genBitAnd(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement bitwise and for {}", .{arch});
    }

    fn genBitOr(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement bitwise or for {}", .{arch});
    }

    fn genXor(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement xor for {}", .{arch});
    }

    fn genOptionalPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement .optional_payload for {}", .{arch});
    }

    fn genOptionalPayloadPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement .optional_payload_ptr for {}", .{arch});
    }

    fn genUnwrapErrErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union error for {}", .{arch});
    }

    fn genUnwrapErrPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union payload for {}", .{arch});
    }
    // *(E!T) -> E
    fn genUnwrapErrErrPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union error ptr for {}", .{arch});
    }
    // *(E!T) -> *T
    fn genUnwrapErrPayloadPtr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement unwrap error union payload ptr for {}", .{arch});
    }
    fn genWrapOptional(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        const optional_ty = inst.base.ty;

        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        // Optional type is just a boolean true
        if (optional_ty.abiSize(self.target.*) == 1)
            return MCValue{ .immediate = 1 };

        return self.fail(inst.base.src, "TODO implement wrap optional for {}", .{arch});
    }

    /// T to E!T
    fn genWrapErrUnionPayload(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return self.fail(inst.base.src, "TODO implement wrap errunion payload for {}", .{arch});
    }

    /// E to E!T
    fn genWrapErrUnionErr(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return self.fail(inst.base.src, "TODO implement wrap errunion error for {}", .{arch});
    }
    fn genVarPtr(self: *Self, inst: *ir.Inst.VarPtr) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;

        return self.fail(inst.base.src, "TODO implement varptr for {}", .{arch});
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
        return try self.genX8664BinMath(&inst.base, inst.lhs, inst.rhs);
    }

    fn genSubWrap(self: *Self, inst: *ir.Inst.BinOp) !MCValue {
        // No side effects, so if it's unreferenced, do nothing.
        if (inst.base.isUnused())
            return MCValue.dead;
        return self.fail(inst.base.src, "TODO implement subwrap for {}", .{arch});
    }

    /// Perform "binary" operators, excluding comparisons.
    /// Currently, the following ops are supported:
    /// ADD, SUB, XOR, OR, AND
    fn genX8664BinMath(self: *Self, inst: *ir.Inst, op_lhs: *ir.Inst, op_rhs: *ir.Inst) !MCValue {
        // We'll handle these ops in two steps.
        // 1) Prepare an output location (register or memory)
        //    This location will be the location of the operand that dies (if one exists)
        //    or just a temporary register (if one doesn't exist)
        // 2) Perform the op with the other argument
        // 3) Sometimes, the output location is memory but the op doesn't support it.
        //    In this case, copy that location to a register, then perform the op to that register instead.
        //
        // TODO: make this algorithm less bad

        try self.code.ensureCapacity(self.code.items.len + 8);

        const lhs = try self.resolveInst(op_lhs);
        const rhs = try self.resolveInst(op_rhs);

        // There are 2 operands, destination and source.
        // Either one, but not both, can be a memory operand.
        // Source operand can be an immediate, 8 bits or 32 bits.
        // So, if either one of the operands dies with this instruction, we can use it
        // as the result MCValue.
        var dst_mcv: MCValue = undefined;
        var src_mcv: MCValue = undefined;
        var src_inst: *ir.Inst = undefined;
        if (self.reuseOperand(inst, 0, lhs)) {
            // LHS dies; use it as the destination.
            // Both operands cannot be memory.
            src_inst = op_rhs;
            if (lhs.isMemory() and rhs.isMemory()) {
                dst_mcv = try self.copyToNewRegister(inst, lhs);
                src_mcv = rhs;
            } else {
                dst_mcv = lhs;
                src_mcv = rhs;
            }
        } else if (self.reuseOperand(inst, 1, rhs)) {
            // RHS dies; use it as the destination.
            // Both operands cannot be memory.
            src_inst = op_lhs;
            if (lhs.isMemory() and rhs.isMemory()) {
                dst_mcv = try self.copyToNewRegister(inst, rhs);
                src_mcv = lhs;
            } else {
                dst_mcv = rhs;
                src_mcv = lhs;
            }
        } else {
            if (lhs.isMemory()) {
                dst_mcv = try self.copyToNewRegister(inst, lhs);
                src_mcv = rhs;
                src_inst = op_rhs;
            } else {
                dst_mcv = try self.copyToNewRegister(inst, rhs);
                src_mcv = lhs;
                src_inst = op_lhs;
            }
        }
        // This instruction supports only signed 32-bit immediates at most. If the immediate
        // value is larger than this, we put it in a register.
        // A potential opportunity for future optimization here would be keeping track
        // of the fact that the instruction is available both as an immediate
        // and as a register.
        switch (src_mcv) {
            .immediate => |imm| {
                if (imm > math.maxInt(u31)) {
                    src_mcv = MCValue{ .register = try self.copyToTmpRegister(src_inst.src, Type.initTag(.u64), src_mcv) };
                }
            },
            else => {},
        }

        // Now for step 2, we perform the actual op
        switch (inst.tag) {
            // TODO: Generate wrapping and non-wrapping versions separately
            .add, .addwrap => try self.genX8664BinMathCode(inst.src, inst.ty, dst_mcv, src_mcv, 0, 0x00),
            .bool_or, .bit_or => try self.genX8664BinMathCode(inst.src, inst.ty, dst_mcv, src_mcv, 1, 0x08),
            .bool_and, .bit_and => try self.genX8664BinMathCode(inst.src, inst.ty, dst_mcv, src_mcv, 4, 0x20),
            .sub, .subwrap => try self.genX8664BinMathCode(inst.src, inst.ty, dst_mcv, src_mcv, 5, 0x28),
            .xor, .not => try self.genX8664BinMathCode(inst.src, inst.ty, dst_mcv, src_mcv, 6, 0x30),

            .mul, .mulwrap => try self.genX8664Imul(inst.src, inst.ty, dst_mcv, src_mcv),
            else => unreachable,
        }

        return dst_mcv;
    }

    /// Wrap over Instruction.encodeInto to translate errors
    fn encodeX8664Instruction(
        self: *Self,
        src: LazySrcLoc,
        inst: Instruction,
    ) !void {
        inst.encodeInto(self.code) catch |err| {
            if (err == error.OutOfMemory)
                return error.OutOfMemory
            else
                return self.fail(src, "Instruction.encodeInto failed because {s}", .{@errorName(err)});
        };
    }

    /// This function encodes a binary operation for x86_64
    /// intended for use with the following opcode ranges
    /// because they share the same structure.
    ///
    /// Thus not all binary operations can be used here
    /// -- multiplication needs to be done with imul,
    /// which doesn't have as convenient an interface.
    ///
    /// "opx"-style instructions use the opcode extension field to indicate which instruction to execute:
    ///
    /// opx = /0: add
    /// opx = /1: or
    /// opx = /2: adc
    /// opx = /3: sbb
    /// opx = /4: and
    /// opx = /5: sub
    /// opx = /6: xor
    /// opx = /7: cmp
    ///
    /// opcode  | operand shape
    /// --------+----------------------
    /// 80 /opx | *r/m8*,        imm8
    /// 81 /opx | *r/m16/32/64*, imm16/32
    /// 83 /opx | *r/m16/32/64*, imm8
    ///
    /// "mr"-style instructions use the low bits of opcode to indicate shape of instruction:
    ///
    /// mr = 00: add
    /// mr = 08: or
    /// mr = 10: adc
    /// mr = 18: sbb
    /// mr = 20: and
    /// mr = 28: sub
    /// mr = 30: xor
    /// mr = 38: cmp
    ///
    /// opcode | operand shape
    /// -------+-------------------------
    /// mr + 0 | *r/m8*,        r8
    /// mr + 1 | *r/m16/32/64*, r16/32/64
    /// mr + 2 | *r8*,          r/m8
    /// mr + 3 | *r16/32/64*,   r/m16/32/64
    /// mr + 4 | *AL*,          imm8
    /// mr + 5 | *rAX*,         imm16/32
    ///
    /// TODO: rotates and shifts share the same structure, so we can potentially implement them
    ///       at a later date with very similar code.
    ///       They have "opx"-style instructions, but no "mr"-style instructions.
    ///
    /// opx = /0: rol,
    /// opx = /1: ror,
    /// opx = /2: rcl,
    /// opx = /3: rcr,
    /// opx = /4: shl sal,
    /// opx = /5: shr,
    /// opx = /6: sal shl,
    /// opx = /7: sar,
    ///
    /// opcode  | operand shape
    /// --------+------------------
    /// c0 /opx | *r/m8*,        imm8
    /// c1 /opx | *r/m16/32/64*, imm8
    /// d0 /opx | *r/m8*,        1
    /// d1 /opx | *r/m16/32/64*, 1
    /// d2 /opx | *r/m8*,        CL    (for context, CL is register 1)
    /// d3 /opx | *r/m16/32/64*, CL    (for context, CL is register 1)
    fn genX8664BinMathCode(
        self: *Self,
        src: LazySrcLoc,
        dst_ty: Type,
        dst_mcv: MCValue,
        src_mcv: MCValue,
        opx: u3,
        mr: u8,
    ) !void {
        switch (dst_mcv) {
            .none => unreachable,
            .undef => unreachable,
            .dead, .unreach, .immediate => unreachable,
            .compare_flags_unsigned => unreachable,
            .compare_flags_signed => unreachable,
            .ptr_stack_offset => unreachable,
            .ptr_embedded_in_code => unreachable,
            .register => |dst_reg| {
                switch (src_mcv) {
                    .none => unreachable,
                    .undef => try self.genSetReg(src, dst_ty, dst_reg, .undef),
                    .dead, .unreach => unreachable,
                    .ptr_stack_offset => unreachable,
                    .ptr_embedded_in_code => unreachable,
                    .register => |src_reg| {
                        // for register, register use mr + 1
                        // addressing mode: *r/m16/32/64*, r16/32/64
                        const abi_size = dst_ty.abiSize(self.target.*);
                        const encoder = try X8664Encoder.init(self.code, 3);
                        encoder.rex(.{
                            .w = abi_size == 8,
                            .r = src_reg.isExtended(),
                            .b = dst_reg.isExtended(),
                        });
                        encoder.opcode_1byte(mr + 1);
                        encoder.modRm_direct(
                            src_reg.low_id(),
                            dst_reg.low_id(),
                        );
                    },
                    .immediate => |imm| {
                        // register, immediate use opx = 81 or 83 addressing modes:
                        // opx = 81: r/m16/32/64, imm16/32
                        // opx = 83: r/m16/32/64, imm8
                        const imm32 = @intCast(i32, imm); // This case must be handled before calling genX8664BinMathCode.
                        if (imm32 <= math.maxInt(i8)) {
                            const abi_size = dst_ty.abiSize(self.target.*);
                            const encoder = try X8664Encoder.init(self.code, 4);
                            encoder.rex(.{
                                .w = abi_size == 8,
                                .b = dst_reg.isExtended(),
                            });
                            encoder.opcode_1byte(0x83);
                            encoder.modRm_direct(
                                opx,
                                dst_reg.low_id(),
                            );
                            encoder.imm8(@intCast(i8, imm32));
                        } else {
                            const abi_size = dst_ty.abiSize(self.target.*);
                            const encoder = try X8664Encoder.init(self.code, 7);
                            encoder.rex(.{
                                .w = abi_size == 8,
                                .b = dst_reg.isExtended(),
                            });
                            encoder.opcode_1byte(0x81);
                            encoder.modRm_direct(
                                opx,
                                dst_reg.low_id(),
                            );
                            encoder.imm32(@intCast(i32, imm32));
                        }
                    },
                    .embedded_in_code, .memory => {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source memory", .{});
                    },
                    .stack_offset => |off| {
                        // register, indirect use mr + 3
                        // addressing mode: *r16/32/64*, r/m16/32/64
                        const abi_size = dst_ty.abiSize(self.target.*);
                        const adj_off = off + abi_size;
                        if (off > math.maxInt(i32)) {
                            return self.fail(src, "stack offset too large", .{});
                        }
                        const encoder = try X8664Encoder.init(self.code, 7);
                        encoder.rex(.{
                            .w = abi_size == 8,
                            .r = dst_reg.isExtended(),
                        });
                        encoder.opcode_1byte(mr + 3);
                        if (adj_off <= std.math.maxInt(i8)) {
                            encoder.modRm_indirectDisp8(
                                dst_reg.low_id(),
                                Register.ebp.low_id(),
                            );
                            encoder.disp8(-@intCast(i8, adj_off));
                        } else {
                            encoder.modRm_indirectDisp32(
                                dst_reg.low_id(),
                                Register.ebp.low_id(),
                            );
                            encoder.disp32(-@intCast(i32, adj_off));
                        }
                    },
                    .compare_flags_unsigned => {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source compare flag (unsigned)", .{});
                    },
                    .compare_flags_signed => {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source compare flag (signed)", .{});
                    },
                }
            },
            .stack_offset => |off| {
                switch (src_mcv) {
                    .none => unreachable,
                    .undef => return self.genSetStack(src, dst_ty, off, .undef),
                    .dead, .unreach => unreachable,
                    .ptr_stack_offset => unreachable,
                    .ptr_embedded_in_code => unreachable,
                    .register => |src_reg| {
                        try self.genX8664ModRMRegToStack(src, dst_ty, off, src_reg, mr + 0x1);
                    },
                    .immediate => |imm| {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source immediate", .{});
                    },
                    .embedded_in_code, .memory, .stack_offset => {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source memory", .{});
                    },
                    .compare_flags_unsigned => {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source compare flag (unsigned)", .{});
                    },
                    .compare_flags_signed => {
                        return self.fail(src, "TODO implement x86 ADD/SUB/CMP source compare flag (signed)", .{});
                    },
                }
            },
            .embedded_in_code, .memory => {
                return self.fail(src, "TODO implement x86 ADD/SUB/CMP destination memory", .{});
            },
        }
    }

    /// Performs integer multiplication between dst_mcv and src_mcv, storing the result in dst_mcv.
    fn genX8664Imul(
        self: *Self,
        src: LazySrcLoc,
        dst_ty: Type,
        dst_mcv: MCValue,
        src_mcv: MCValue,
    ) !void {
        switch (dst_mcv) {
            .none => unreachable,
            .undef => unreachable,
            .dead, .unreach, .immediate => unreachable,
            .compare_flags_unsigned => unreachable,
            .compare_flags_signed => unreachable,
            .ptr_stack_offset => unreachable,
            .ptr_embedded_in_code => unreachable,
            .register => |dst_reg| {
                switch (src_mcv) {
                    .none => unreachable,
                    .undef => try self.genSetReg(src, dst_ty, dst_reg, .undef),
                    .dead, .unreach => unreachable,
                    .ptr_stack_offset => unreachable,
                    .ptr_embedded_in_code => unreachable,
                    .register => |src_reg| {
                        // register, register
                        //
                        // Use the following imul opcode
                        // 0F AF /r: IMUL r32/64, r/m32/64
                        const abi_size = dst_ty.abiSize(self.target.*);
                        const encoder = try X8664Encoder.init(self.code, 4);
                        encoder.rex(.{
                            .w = abi_size == 8,
                            .r = dst_reg.isExtended(),
                            .b = src_reg.isExtended(),
                        });
                        encoder.opcode_2byte(0x0f, 0xaf);
                        encoder.modRm_direct(
                            dst_reg.low_id(),
                            src_reg.low_id(),
                        );
                    },
                    .immediate => |imm| {
                        // register, immediate:
                        // depends on size of immediate.
                        //
                        // immediate fits in i8:
                        // 6B /r ib: IMUL r32/64, r/m32/64, imm8
                        //
                        // immediate fits in i32:
                        // 69 /r id: IMUL r32/64, r/m32/64, imm32
                        //
                        // immediate is huge:
                        // split into 2 instructions
                        // 1) copy the 64 bit immediate into a tmp register
                        // 2) perform register,register mul
                        // 0F AF /r: IMUL r32/64, r/m32/64
                        if (math.minInt(i8) <= imm and imm <= math.maxInt(i8)) {
                            const abi_size = dst_ty.abiSize(self.target.*);
                            const encoder = try X8664Encoder.init(self.code, 4);
                            encoder.rex(.{
                                .w = abi_size == 8,
                                .r = dst_reg.isExtended(),
                                .b = dst_reg.isExtended(),
                            });
                            encoder.opcode_1byte(0x6B);
                            encoder.modRm_direct(
                                dst_reg.low_id(),
                                dst_reg.low_id(),
                            );
                            encoder.imm8(@intCast(i8, imm));
                        } else if (math.minInt(i32) <= imm and imm <= math.maxInt(i32)) {
                            const abi_size = dst_ty.abiSize(self.target.*);
                            const encoder = try X8664Encoder.init(self.code, 7);
                            encoder.rex(.{
                                .w = abi_size == 8,
                                .r = dst_reg.isExtended(),
                                .b = dst_reg.isExtended(),
                            });
                            encoder.opcode_1byte(0x69);
                            encoder.modRm_direct(
                                dst_reg.low_id(),
                                dst_reg.low_id(),
                            );
                            encoder.imm32(@intCast(i32, imm));
                        } else {
                            const src_reg = try self.copyToTmpRegister(src, dst_ty, src_mcv);
                            return self.genX8664Imul(src, dst_ty, dst_mcv, MCValue{ .register = src_reg });
                        }
                    },
                    .embedded_in_code, .memory, .stack_offset => {
                        return self.fail(src, "TODO implement x86 multiply source memory", .{});
                    },
                    .compare_flags_unsigned => {
                        return self.fail(src, "TODO implement x86 multiply source compare flag (unsigned)", .{});
                    },
                    .compare_flags_signed => {
                        return self.fail(src, "TODO implement x86 multiply source compare flag (signed)", .{});
                    },
                }
            },
            .stack_offset => |off| {
                switch (src_mcv) {
                    .none => unreachable,
                    .undef => return self.genSetStack(src, dst_ty, off, .undef),
                    .dead, .unreach => unreachable,
                    .ptr_stack_offset => unreachable,
                    .ptr_embedded_in_code => unreachable,
                    .register => |src_reg| {
                        // copy dst to a register
                        const dst_reg = try self.copyToTmpRegister(src, dst_ty, dst_mcv);
                        // multiply into dst_reg
                        // register, register
                        // Use the following imul opcode
                        // 0F AF /r: IMUL r32/64, r/m32/64
                        const abi_size = dst_ty.abiSize(self.target.*);
                        const encoder = try X8664Encoder.init(self.code, 4);
                        encoder.rex(.{
                            .w = abi_size == 8,
                            .r = dst_reg.isExtended(),
                            .b = src_reg.isExtended(),
                        });
                        encoder.opcode_2byte(0x0f, 0xaf);
                        encoder.modRm_direct(
                            dst_reg.low_id(),
                            src_reg.low_id(),
                        );
                        // copy dst_reg back out
                        return self.genSetStack(src, dst_ty, off, MCValue{ .register = dst_reg });
                    },
                    .immediate => |imm| {
                        return self.fail(src, "TODO implement x86 multiply source immediate", .{});
                    },
                    .embedded_in_code, .memory, .stack_offset => {
                        return self.fail(src, "TODO implement x86 multiply source memory", .{});
                    },
                    .compare_flags_unsigned => {
                        return self.fail(src, "TODO implement x86 multiply source compare flag (unsigned)", .{});
                    },
                    .compare_flags_signed => {
                        return self.fail(src, "TODO implement x86 multiply source compare flag (signed)", .{});
                    },
                }
            },
            .embedded_in_code, .memory => {
                return self.fail(src, "TODO implement x86 multiply destination memory", .{});
            },
        }
    }

    fn genX8664ModRMRegToStack(self: *Self, src: LazySrcLoc, ty: Type, off: u32, reg: Register, opcode: u8) !void {
        const abi_size = ty.abiSize(self.target.*);
        const adj_off = off + abi_size;
        if (off > math.maxInt(i32)) {
            return self.fail(src, "stack offset too large", .{});
        }

        const i_adj_off = -@intCast(i32, adj_off);
        const encoder = try X8664Encoder.init(self.code, 7);
        encoder.rex(.{
            .w = abi_size == 8,
            .r = reg.isExtended(),
        });
        encoder.opcode_1byte(opcode);
        if (i_adj_off < std.math.maxInt(i8)) {
            // example: 48 89 55 7f           mov    QWORD PTR [rbp+0x7f],rdx
            encoder.modRm_indirectDisp8(
                reg.low_id(),
                Register.ebp.low_id(),
            );
            encoder.disp8(@intCast(i8, i_adj_off));
        } else {
            // example: 48 89 95 80 00 00 00  mov    QWORD PTR [rbp+0x80],rdx
            encoder.modRm_indirectDisp32(
                reg.low_id(),
                Register.ebp.low_id(),
            );
            encoder.disp32(i_adj_off);
        }
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
            else => {},
        }
    }

    fn genArg(self: *Self, inst: *ir.Inst.Arg) !MCValue {
        const arg_index = self.arg_index;
        self.arg_index += 1;

        if (callee_preserved_regs.len == 0) {
            return self.fail(inst.base.src, "TODO implement Register enum for {}", .{arch});
        }

        const result = self.args[arg_index];
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
        try self.code.append(0xcc); // int3
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
                const arg_mcv = try self.resolveInst(inst.args[arg_i]);
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
                        return self.fail(inst.base.src, "TODO implement calling with parameters in memory", .{});
                    },
                    .ptr_stack_offset => {
                        return self.fail(inst.base.src, "TODO implement calling with MCValue.ptr_stack_offset arg", .{});
                    },
                    .ptr_embedded_in_code => {
                        return self.fail(inst.base.src, "TODO implement calling with MCValue.ptr_embedded_in_code arg", .{});
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

                    const ptr_bits = arch.ptrBitWidth();
                    const ptr_bytes: u64 = @divExact(ptr_bits, 8);
                    const got_addr = if (self.bin_file.cast(link.File.Elf)) |elf_file| blk: {
                        const got = &elf_file.program_headers.items[elf_file.phdr_got_index.?];
                        break :blk @intCast(u32, got.p_vaddr + func.owner_decl.link.elf.offset_table_index * ptr_bytes);
                    } else if (self.bin_file.cast(link.File.Coff)) |coff_file|
                        @intCast(u32, coff_file.offset_table_virtual_address + func.owner_decl.link.coff.offset_table_index * ptr_bytes)
                    else
                        unreachable;

                    // ff 14 25 xx xx xx xx    call [addr]
                    try self.code.ensureCapacity(self.code.items.len + 7);
                    self.code.appendSliceAssumeCapacity(&[3]u8{ 0xff, 0x14, 0x25 });
                    mem.writeIntLittle(u32, self.code.addManyAsArrayAssumeCapacity(4), got_addr);
                } else if (func_value.castTag(.extern_fn)) |_| {
                    return self.fail(inst.base.src, "TODO implement calling extern functions", .{});
                } else {
                    return self.fail(inst.base.src, "TODO implement calling bitcasted functions", .{});
                }
            } else {
                return self.fail(inst.base.src, "TODO implement calling runtime known function pointer", .{});
            }
        } else if (self.bin_file.cast(link.File.MachO)) |macho_file| {
            for (info.args) |mc_arg, arg_i| {
                const arg = inst.args[arg_i];
                const arg_mcv = try self.resolveInst(inst.args[arg_i]);
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
                        return self.fail(inst.base.src, "TODO implement calling with parameters in memory", .{});
                    },
                    .ptr_stack_offset => {
                        return self.fail(inst.base.src, "TODO implement calling with MCValue.ptr_stack_offset arg", .{});
                    },
                    .ptr_embedded_in_code => {
                        return self.fail(inst.base.src, "TODO implement calling with MCValue.ptr_embedded_in_code arg", .{});
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
                    try self.genSetReg(inst.base.src, Type.initTag(.u64), .rax, .{ .memory = got_addr });
                    // callq *%rax
                    try self.code.ensureCapacity(self.code.items.len + 2);
                    self.code.appendSliceAssumeCapacity(&[2]u8{ 0xff, 0xd0 });
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

                    // callq
                    try self.code.ensureCapacity(self.code.items.len + 5);
                    self.code.appendSliceAssumeCapacity(&[5]u8{ 0xe8, 0x0, 0x0, 0x0, 0x0 });

                    try macho_file.stub_fixups.append(self.bin_file.allocator, .{
                        .symbol = symbol,
                        .already_defined = already_defined,
                        .start = start,
                        .len = 5,
                    });
                    // We mark the space and fix it up later.
                } else {
                    return self.fail(inst.base.src, "TODO implement calling bitcasted functions", .{});
                }
            } else {
                return self.fail(inst.base.src, "TODO implement calling runtime known function pointer", .{});
            }
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

        // TODO when implementing defer, this will need to jump to the appropriate defer expression.
        // TODO optimization opportunity: figure out when we can emit this as a 2 byte instruction
        // which is available if the jump is 127 bytes or less forward.
        try self.code.resize(self.code.items.len + 5);
        self.code.items[self.code.items.len - 5] = 0xe9; // jmp rel32
        try self.exitlude_jump_relocs.append(self.gpa, self.code.items.len - 4);

        return .unreach;
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

        try self.code.ensureCapacity(self.code.items.len + 8);

        const lhs = try self.resolveInst(inst.lhs);
        const rhs = try self.resolveInst(inst.rhs);

        // There are 2 operands, destination and source.
        // Either one, but not both, can be a memory operand.
        // Source operand can be an immediate, 8 bits or 32 bits.
        const dst_mcv = if (lhs.isImmediate() or (lhs.isMemory() and rhs.isMemory()))
            try self.copyToNewRegister(&inst.base, lhs)
        else
            lhs;
        // This instruction supports only signed 32-bit immediates at most.
        const src_mcv = try self.limitImmediateType(inst.rhs, i32);

        try self.genX8664BinMathCode(inst.base.src, inst.base.ty, dst_mcv, src_mcv, 7, 0x38);
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
        try self.dbgAdvancePCAndLine(inst.byte_offset);
        assert(inst.base.isUnused());
        return MCValue.dead;
    }

    fn genCondBr(self: *Self, inst: *ir.Inst.CondBr) !MCValue {
        const cond = try self.resolveInst(inst.condition);

        const reloc: Reloc = reloc: {
            try self.code.ensureCapacity(self.code.items.len + 6);

            const opcode: u8 = switch (cond) {
                .compare_flags_signed => |cmp_op| blk: {
                    // Here we map to the opposite opcode because the jump is to the false branch.
                    const opcode: u8 = switch (cmp_op) {
                        .gte => 0x8c,
                        .gt => 0x8e,
                        .neq => 0x84,
                        .lt => 0x8d,
                        .lte => 0x8f,
                        .eq => 0x85,
                    };
                    break :blk opcode;
                },
                .compare_flags_unsigned => |cmp_op| blk: {
                    // Here we map to the opposite opcode because the jump is to the false branch.
                    const opcode: u8 = switch (cmp_op) {
                        .gte => 0x82,
                        .gt => 0x86,
                        .neq => 0x84,
                        .lt => 0x83,
                        .lte => 0x87,
                        .eq => 0x85,
                    };
                    break :blk opcode;
                },
                .register => |reg| blk: {
                    // test reg, 1
                    // TODO detect al, ax, eax
                    const encoder = try X8664Encoder.init(self.code, 4);
                    encoder.rex(.{
                        // TODO audit this codegen: we force w = true here to make
                        // the value affect the big register
                        .w = true,
                        .b = reg.isExtended(),
                    });
                    encoder.opcode_1byte(0xf6);
                    encoder.modRm_direct(
                        0,
                        reg.low_id(),
                    );
                    encoder.disp8(1);
                    break :blk 0x84;
                },
                else => return self.fail(inst.base.src, "TODO implement condbr {s} when condition is {s}", .{ arch, @tagName(cond) }),
            };
            self.code.appendSliceAssumeCapacity(&[_]u8{ 0x0f, opcode });
            const reloc = Reloc{ .rel32 = self.code.items.len };
            self.code.items.len += 4;
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
        try self.ensureProcessDeathCapacity(then_deaths.len);
        for (then_deaths) |operand| {
            self.processDeath(operand);
        }
        try self.genBody(inst.then_body);

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
        try self.ensureProcessDeathCapacity(else_deaths.len);
        for (else_deaths) |operand| {
            self.processDeath(operand);
        }
        try self.genBody(inst.else_body);

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
            try self.setRegOrMem(inst.base.src, else_entry.key.ty, canon_mcv, else_entry.value);
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
            try self.setRegOrMem(inst.base.src, then_entry.key.ty, parent_mcv, then_entry.value);
            // TODO track the new register / stack allocation
        }

        self.branch_stack.pop().deinit(self.gpa);

        return MCValue.unreach;
    }

    fn genIsNull(self: *Self, inst: *ir.Inst.UnOp) !MCValue {
        return self.fail(inst.base.src, "TODO implement isnull for {}", .{arch});
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
        return self.fail(inst.base.src, "TODO implement iserr for {}", .{arch});
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
        try self.code.ensureCapacity(self.code.items.len + 5);
        if (math.cast(i8, @intCast(i32, index) - (@intCast(i32, self.code.items.len + 2)))) |delta| {
            self.code.appendAssumeCapacity(0xeb); // jmp rel8
            self.code.appendAssumeCapacity(@bitCast(u8, delta));
        } else |_| {
            const delta = @intCast(i32, index) - (@intCast(i32, self.code.items.len + 5));
            self.code.appendAssumeCapacity(0xe9); // jmp rel32
            mem.writeIntLittle(i32, self.code.addManyAsArrayAssumeCapacity(4), delta);
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

        try self.genBody(inst.body);

        for (inst.codegen.relocs.items) |reloc| try self.performReloc(inst.base.src, reloc);

        return @bitCast(MCValue, inst.codegen.mcv);
    }

    fn genSwitch(self: *Self, inst: *ir.Inst.SwitchBr) !MCValue {
        return self.fail(inst.base.src, "TODO genSwitch for {}", .{arch});
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
            .arm_branch => unreachable, // attempting to perfrom an ARM relocation on a non-ARM target arch
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
        switch (inst.base.tag) {
            // lhs AND rhs
            .bool_and => return try self.genX8664BinMath(&inst.base, inst.lhs, inst.rhs),
            // lhs OR rhs
            .bool_or => return try self.genX8664BinMath(&inst.base, inst.lhs, inst.rhs),
            else => unreachable, // Not a boolean operation
        }
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

        // TODO optimization opportunity: figure out when we can emit this as a 2 byte instruction
        // which is available if the jump is 127 bytes or less forward.
        try self.code.resize(self.code.items.len + 5);
        self.code.items[self.code.items.len - 5] = 0xe9; // jmp rel32
        // Leave the jump offset undefined
        block.codegen.relocs.appendAssumeCapacity(.{ .rel32 = self.code.items.len - 4 });

        return .none;
    }

    fn genAsm(self: *Self, inst: *ir.Inst.Assembly) !MCValue {
        if (!inst.is_volatile and inst.base.isUnused())
            return MCValue.dead;

        for (inst.inputs) |input, i| {
            if (input.len < 3 or input[0] != '{' or input[input.len - 1] != '}') {
                return self.fail(inst.base.src, "unrecognized asm input constraint: '{s}'", .{input});
            }
            const reg_name = input[1 .. input.len - 1];
            const reg = parseRegName(reg_name) orelse
                return self.fail(inst.base.src, "unrecognized register: '{s}'", .{reg_name});

            const arg = inst.args[i];
            const arg_mcv = try self.resolveInst(arg);
            try self.register_manager.getRegWithoutTracking(reg);
            try self.genSetReg(inst.base.src, arg.ty, reg, arg_mcv);
        }

        if (mem.eql(u8, inst.asm_source, "syscall")) {
            try self.code.appendSlice(&[_]u8{ 0x0f, 0x05 });
        } else if (inst.asm_source.len != 0) {
            return self.fail(inst.base.src, "TODO implement support for more x86 assembly instructions", .{});
        }

        if (inst.output_name) |output| {
            if (output.len < 4 or output[0] != '=' or output[1] != '{' or output[output.len - 1] != '}') {
                return self.fail(inst.base.src, "unrecognized asm output constraint: '{s}'", .{output});
            }
            const reg_name = output[2 .. output.len - 1];
            const reg = parseRegName(reg_name) orelse
                return self.fail(inst.base.src, "unrecognized register: '{s}'", .{reg_name});
            return MCValue{ .register = reg };
        } else {
            return MCValue.none;
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
                    else => return self.fail(src, "TODO implement memset", .{}),
                }
            },
            .compare_flags_unsigned => |op| {
                return self.fail(src, "TODO implement set stack variable with compare flags value (unsigned)", .{});
            },
            .compare_flags_signed => |op| {
                return self.fail(src, "TODO implement set stack variable with compare flags value (signed)", .{});
            },
            .immediate => |x_big| {
                const abi_size = ty.abiSize(self.target.*);
                const adj_off = stack_offset + abi_size;
                if (adj_off > 128) {
                    return self.fail(src, "TODO implement set stack variable with large stack offset", .{});
                }
                try self.code.ensureCapacity(self.code.items.len + 8);
                switch (abi_size) {
                    1 => {
                        return self.fail(src, "TODO implement set abi_size=1 stack variable with immediate", .{});
                    },
                    2 => {
                        return self.fail(src, "TODO implement set abi_size=2 stack variable with immediate", .{});
                    },
                    4 => {
                        const x = @intCast(u32, x_big);
                        // We have a positive stack offset value but we want a twos complement negative
                        // offset from rbp, which is at the top of the stack frame.
                        const negative_offset = @intCast(i8, -@intCast(i32, adj_off));
                        const twos_comp = @bitCast(u8, negative_offset);
                        // mov    DWORD PTR [rbp+offset], immediate
                        self.code.appendSliceAssumeCapacity(&[_]u8{ 0xc7, 0x45, twos_comp });
                        mem.writeIntLittle(u32, self.code.addManyAsArrayAssumeCapacity(4), x);
                    },
                    8 => {
                        // We have a positive stack offset value but we want a twos complement negative
                        // offset from rbp, which is at the top of the stack frame.
                        const negative_offset = @intCast(i8, -@intCast(i32, adj_off));
                        const twos_comp = @bitCast(u8, negative_offset);

                        // 64 bit write to memory would take two mov's anyways so we
                        // insted just use two 32 bit writes to avoid register allocation
                        try self.code.ensureCapacity(self.code.items.len + 14);
                        var buf: [8]u8 = undefined;
                        mem.writeIntLittle(u64, &buf, x_big);

                        // mov    DWORD PTR [rbp+offset+4], immediate
                        self.code.appendSliceAssumeCapacity(&[_]u8{ 0xc7, 0x45, twos_comp + 4 });
                        self.code.appendSliceAssumeCapacity(buf[4..8]);

                        // mov    DWORD PTR [rbp+offset], immediate
                        self.code.appendSliceAssumeCapacity(&[_]u8{ 0xc7, 0x45, twos_comp });
                        self.code.appendSliceAssumeCapacity(buf[0..4]);
                    },
                    else => {
                        return self.fail(src, "TODO implement set abi_size=large stack variable with immediate", .{});
                    },
                }
            },
            .embedded_in_code => |code_offset| {
                return self.fail(src, "TODO implement set stack variable from embedded_in_code", .{});
            },
            .register => |reg| {
                try self.genX8664ModRMRegToStack(src, ty, stack_offset, reg, 0x89);
            },
            .memory => |vaddr| {
                return self.fail(src, "TODO implement set stack variable from memory vaddr", .{});
            },
            .stack_offset => |off| {
                if (stack_offset == off)
                    return; // Copy stack variable to itself; nothing to do.

                const reg = try self.copyToTmpRegister(src, ty, mcv);
                return self.genSetStack(src, ty, stack_offset, MCValue{ .register = reg });
            },
        }
    }

    fn genSetReg(self: *Self, src: LazySrcLoc, ty: Type, reg: Register, mcv: MCValue) InnerError!void {
        switch (mcv) {
            .dead => unreachable,
            .ptr_stack_offset => unreachable,
            .ptr_embedded_in_code => unreachable,
            .unreach, .none => return, // Nothing to do.
            .undef => {
                if (!self.wantSafety())
                    return; // The already existing value will do just fine.
                // Write the debug undefined value.
                switch (reg.size()) {
                    8 => return self.genSetReg(src, ty, reg, .{ .immediate = 0xaa }),
                    16 => return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaa }),
                    32 => return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaaaaaa }),
                    64 => return self.genSetReg(src, ty, reg, .{ .immediate = 0xaaaaaaaaaaaaaaaa }),
                    else => unreachable,
                }
            },
            .compare_flags_unsigned => |op| {
                const encoder = try X8664Encoder.init(self.code, 7);
                // TODO audit this codegen: we force w = true here to make
                // the value affect the big register
                encoder.rex(.{
                    .w = true,
                    .b = reg.isExtended(),
                });
                encoder.opcode_2byte(0x0f, switch (op) {
                    .gte => 0x93,
                    .gt => 0x97,
                    .neq => 0x95,
                    .lt => 0x92,
                    .lte => 0x96,
                    .eq => 0x94,
                });
                encoder.modRm_direct(
                    0,
                    reg.low_id(),
                );
            },
            .compare_flags_signed => |op| {
                return self.fail(src, "TODO set register with compare flags value (signed)", .{});
            },
            .immediate => |x| {
                // 32-bit moves zero-extend to 64-bit, so xoring the 32-bit
                // register is the fastest way to zero a register.
                if (x == 0) {
                    // The encoding for `xor r32, r32` is `0x31 /r`.
                    const encoder = try X8664Encoder.init(self.code, 3);

                    // If we're accessing e.g. r8d, we need to use a REX prefix before the actual operation. Since
                    // this is a 32-bit operation, the W flag is set to zero. X is also zero, as we're not using a SIB.
                    // Both R and B are set, as we're extending, in effect, the register bits *and* the operand.
                    encoder.rex(.{
                        .r = reg.isExtended(),
                        .b = reg.isExtended(),
                    });
                    encoder.opcode_1byte(0x31);
                    // Section 3.1.1.1 of the Intel x64 Manual states that "/r indicates that the
                    // ModR/M byte of the instruction contains a register operand and an r/m operand."
                    encoder.modRm_direct(
                        reg.low_id(),
                        reg.low_id(),
                    );

                    return;
                }
                if (x <= math.maxInt(i32)) {
                    // Next best case: if we set the lower four bytes, the upper four will be zeroed.
                    //
                    // The encoding for `mov IMM32 -> REG` is (0xB8 + R) IMM.

                    const encoder = try X8664Encoder.init(self.code, 6);
                    // Just as with XORing, we need a REX prefix. This time though, we only
                    // need the B bit set, as we're extending the opcode's register field,
                    // and there is no Mod R/M byte.
                    encoder.rex(.{
                        .b = reg.isExtended(),
                    });
                    encoder.opcode_withReg(0xB8, reg.low_id());

                    // no ModR/M byte

                    // IMM
                    encoder.imm32(@intCast(i32, x));
                    return;
                }
                // Worst case: we need to load the 64-bit register with the IMM. GNU's assemblers calls
                // this `movabs`, though this is officially just a different variant of the plain `mov`
                // instruction.
                //
                // This encoding is, in fact, the *same* as the one used for 32-bit loads. The only
                // difference is that we set REX.W before the instruction, which extends the load to
                // 64-bit and uses the full bit-width of the register.
                {
                    const encoder = try X8664Encoder.init(self.code, 10);
                    encoder.rex(.{
                        .w = true,
                        .b = reg.isExtended(),
                    });
                    encoder.opcode_withReg(0xB8, reg.low_id());
                    encoder.imm64(x);
                }
            },
            .embedded_in_code => |code_offset| {
                // We need the offset from RIP in a signed i32 twos complement.
                // The instruction is 7 bytes long and RIP points to the next instruction.

                // 64-bit LEA is encoded as REX.W 8D /r.
                const rip = self.code.items.len + 7;
                const big_offset = @intCast(i64, code_offset) - @intCast(i64, rip);
                const offset = @intCast(i32, big_offset);
                const encoder = try X8664Encoder.init(self.code, 7);

                // byte 1, always exists because w = true
                encoder.rex(.{
                    .w = true,
                    .r = reg.isExtended(),
                });
                // byte 2
                encoder.opcode_1byte(0x8D);
                // byte 3
                encoder.modRm_RIPDisp32(reg.low_id());
                // byte 4-7
                encoder.disp32(offset);

                // Double check that we haven't done any math errors
                assert(rip == self.code.items.len);
            },
            .register => |src_reg| {
                // If the registers are the same, nothing to do.
                if (src_reg.id() == reg.id())
                    return;

                // This is a variant of 8B /r.
                const abi_size = ty.abiSize(self.target.*);
                const encoder = try X8664Encoder.init(self.code, 3);
                encoder.rex(.{
                    .w = abi_size == 8,
                    .r = reg.isExtended(),
                    .b = src_reg.isExtended(),
                });
                encoder.opcode_1byte(0x8B);
                encoder.modRm_direct(reg.low_id(), src_reg.low_id());
            },
            .memory => |x| {
                if (self.bin_file.options.pie) {
                    // RIP-relative displacement to the entry in the GOT table.
                    const abi_size = ty.abiSize(self.target.*);
                    const encoder = try X8664Encoder.init(self.code, 10);

                    // LEA reg, [<offset>]

                    // We encode the instruction FIRST because prefixes may or may not appear.
                    // After we encode the instruction, we will know that the displacement bytes
                    // for [<offset>] will be at self.code.items.len - 4.
                    encoder.rex(.{
                        .w = true, // force 64 bit because loading an address (to the GOT)
                        .r = reg.isExtended(),
                    });
                    encoder.opcode_1byte(0x8D);
                    encoder.modRm_RIPDisp32(reg.low_id());
                    encoder.disp32(0);

                    // TODO we should come up with our own, backend independent relocation types
                    // which each backend (Elf, MachO, etc.) would then translate into an actual
                    // fixup when linking.
                    if (self.bin_file.cast(link.File.MachO)) |macho_file| {
                        try macho_file.pie_fixups.append(self.bin_file.allocator, .{
                            .target_addr = x,
                            .offset = self.code.items.len - 4,
                            .size = 4,
                        });
                    } else {
                        return self.fail(src, "TODO implement genSetReg for PIE GOT indirection on this platform", .{});
                    }

                    // MOV reg, [reg]
                    encoder.rex(.{
                        .w = abi_size == 8,
                        .r = reg.isExtended(),
                        .b = reg.isExtended(),
                    });
                    encoder.opcode_1byte(0x8B);
                    encoder.modRm_indirectDisp0(reg.low_id(), reg.low_id());
                } else if (x <= math.maxInt(i32)) {
                    // Moving from memory to a register is a variant of `8B /r`.
                    // Since we're using 64-bit moves, we require a REX.
                    // This variant also requires a SIB, as it would otherwise be RIP-relative.
                    // We want mode zero with the lower three bits set to four to indicate an SIB with no other displacement.
                    // The SIB must be 0x25, to indicate a disp32 with no scaled index.
                    // 0b00RRR100, where RRR is the lower three bits of the register ID.
                    // The instruction is thus eight bytes; REX 0x8B 0b00RRR100 0x25 followed by a four-byte disp32.
                    const abi_size = ty.abiSize(self.target.*);
                    const encoder = try X8664Encoder.init(self.code, 8);
                    encoder.rex(.{
                        .w = abi_size == 8,
                        .r = reg.isExtended(),
                    });
                    encoder.opcode_1byte(0x8B);
                    // effective address = [SIB]
                    encoder.modRm_SIBDisp0(reg.low_id());
                    // SIB = disp32
                    encoder.sib_disp32();
                    encoder.disp32(@intCast(i32, x));
                } else {
                    // If this is RAX, we can use a direct load; otherwise, we need to load the address, then indirectly load
                    // the value.
                    if (reg.id() == 0) {
                        // REX.W 0xA1 moffs64*
                        // moffs64* is a 64-bit offset "relative to segment base", which really just means the
                        // absolute address for all practical purposes.

                        const encoder = try X8664Encoder.init(self.code, 10);
                        encoder.rex(.{
                            .w = true,
                        });
                        encoder.opcode_1byte(0xA1);
                        encoder.writeIntLittle(u64, x);
                    } else {
                        // This requires two instructions; a move imm as used above, followed by an indirect load using the register
                        // as the address and the register as the destination.
                        //
                        // This cannot be used if the lower three bits of the id are equal to four or five, as there
                        // is no way to possibly encode it. This means that RSP, RBP, R12, and R13 cannot be used with
                        // this instruction.
                        const id3 = @truncate(u3, reg.id());
                        assert(id3 != 4 and id3 != 5);

                        // Rather than duplicate the logic used for the move, we just use a self-call with a new MCValue.
                        try self.genSetReg(src, ty, reg, MCValue{ .immediate = x });

                        // Now, the register contains the address of the value to load into it
                        // Currently, we're only allowing 64-bit registers, so we need the `REX.W 8B /r` variant.
                        // TODO: determine whether to allow other sized registers, and if so, handle them properly.

                        // mov reg, [reg]
                        const abi_size = ty.abiSize(self.target.*);
                        const encoder = try X8664Encoder.init(self.code, 3);
                        encoder.rex(.{
                            .w = abi_size == 8,
                            .r = reg.isExtended(),
                            .b = reg.isExtended(),
                        });
                        encoder.opcode_1byte(0x8B);
                        encoder.modRm_indirectDisp0(reg.low_id(), reg.low_id());
                    }
                }
            },
            .stack_offset => |unadjusted_off| {
                const abi_size = ty.abiSize(self.target.*);
                const off = unadjusted_off + abi_size;
                if (off < std.math.minInt(i32) or off > std.math.maxInt(i32)) {
                    return self.fail(src, "stack offset too large", .{});
                }
                const ioff = -@intCast(i32, off);
                const encoder = try X8664Encoder.init(self.code, 3);
                encoder.rex(.{
                    .w = abi_size == 8,
                    .r = reg.isExtended(),
                });
                encoder.opcode_1byte(0x8B);
                if (std.math.minInt(i8) <= ioff and ioff <= std.math.maxInt(i8)) {
                    // Example: 48 8b 4d 7f           mov    rcx,QWORD PTR [rbp+0x7f]
                    encoder.modRm_indirectDisp8(reg.low_id(), Register.ebp.low_id());
                    encoder.disp8(@intCast(i8, ioff));
                } else {
                    // Example: 48 8b 8d 80 00 00 00  mov    rcx,QWORD PTR [rbp+0x80]
                    encoder.modRm_indirectDisp32(reg.low_id(), Register.ebp.low_id());
                    encoder.disp32(ioff);
                }
            },
        }
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
        const ptr_bits = arch.ptrBitWidth();
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

        switch (cc) {
            .Naked => {
                assert(result.args.len == 0);
                result.return_value = .{ .unreach = {} };
                result.stack_byte_count = 0;
                result.stack_align = 1;
                return result;
            },
            .Unspecified, .C => {
                var next_int_reg: usize = 0;
                var next_stack_offset: u32 = 0;

                for (param_types) |ty, i| {
                    switch (ty.zigTypeTag()) {
                        .Bool, .Int => {
                            if (!ty.hasCodeGenBits()) {
                                assert(cc != .C);
                                result.args[i] = .{ .none = {} };
                            } else {
                                const param_size = @intCast(u32, ty.abiSize(self.target.*));
                                if (next_int_reg >= c_abi_int_param_regs.len) {
                                    result.args[i] = .{ .stack_offset = next_stack_offset };
                                    next_stack_offset += param_size;
                                } else {
                                    const aliased_reg = registerAlias(
                                        c_abi_int_param_regs[next_int_reg],
                                        param_size,
                                    );
                                    result.args[i] = .{ .register = aliased_reg };
                                    next_int_reg += 1;
                                }
                            }
                        },
                        else => return self.fail(src, "TODO implement function parameters of type {s}", .{@tagName(ty.zigTypeTag())}),
                    }
                }
                result.stack_byte_count = next_stack_offset;
                result.stack_align = 16;
            },
            else => return self.fail(src, "TODO implement function parameters for {} on x86_64", .{cc}),
        }

        if (ret_ty.zigTypeTag() == .NoReturn) {
            result.return_value = .{ .unreach = {} };
        } else if (!ret_ty.hasCodeGenBits()) {
            result.return_value = .{ .none = {} };
        } else switch (cc) {
            .Naked => unreachable,
            .Unspecified, .C => {
                const ret_ty_size = @intCast(u32, ret_ty.abiSize(self.target.*));
                const aliased_reg = registerAlias(c_abi_int_return_regs[0], ret_ty_size);
                result.return_value = .{ .register = aliased_reg };
            },
            else => return self.fail(src, "TODO implement function return values for {}", .{cc}),
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
        // For x86_64 we have to pick a smaller register alias depending on abi size.
        switch (size_bytes) {
            1 => return reg.to8(),
            2 => return reg.to16(),
            4 => return reg.to32(),
            8 => return reg.to64(),
            else => unreachable,
        }
    }

    /// For most architectures this does nothing. For x86_64 it resolves any aliased registers
    /// to the 64-bit wide ones.
    fn toCanonicalReg(reg: Register) Register {
        return reg.to64();
    }
};
