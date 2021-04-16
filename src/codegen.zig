const std = @import("std");
const mem = std.mem;
const math = std.math;
const assert = std.debug.assert;
const ir = @import("ir.zig");
const Type = @import("type.zig").Type;
const Value = @import("value.zig").Value;
const TypedValue = @import("TypedValue.zig");
const link = @import("link.zig");
const Module = @import("Module.zig");
const Compilation = @import("Compilation.zig");
const ErrorMsg = Module.ErrorMsg;
const Target = std.Target;
const Allocator = mem.Allocator;
const trace = @import("tracy.zig").trace;
const DW = std.dwarf;
const leb128 = std.leb;
const log = std.log.scoped(.codegen);
const build_options = @import("build_options");
const LazySrcLoc = Module.LazySrcLoc;
const RegisterManager = @import("register_manager.zig").RegisterManager;

const X8664 = @import("codegen/x86_64.zig");
const X8664Encoder = X8664.Encoder;
const X8664Function = X8664.Function;

const Arm = @import("codegen/arm.zig");
const ArmFunction = Arm.Function;

const AArch64 = @import("codegen/aarch64.zig");
const AArch64Function = AArch64.Function;

const RV64 = @import("codegen/riscv64.zig");
const RV64Function = RV64.Function;

const SPU_2 = @import("codegen/spu-mk2.zig");
const SPU_2Function = SPU_2.Function;

/// The codegen-related data that is stored in `ir.Inst.Block` instructions.
pub const BlockData = struct {
    relocs: std.ArrayListUnmanaged(Reloc) = undefined,
    /// The first break instruction encounters `null` here and chooses a
    /// machine code value for the block result, populating this field.
    /// Following break instructions encounter that value and use it for
    /// the location to store their block results.
    mcv: AnyMCValue = undefined,
};

/// Architecture-independent MCValue. Here, we have a type that is the same size as
/// the architecture-specific MCValue. Next to the declaration of MCValue is a
/// comptime assert that makes sure we guessed correctly about the size. This only
/// exists so that we can bitcast an arch-independent field to and from the real MCValue.
pub const AnyMCValue = extern struct {
    a: usize,
    b: u64,
};

pub const Reloc = union(enum) {
    /// The value is an offset into the `Function` `code` from the beginning.
    /// To perform the reloc, write 32-bit signed little-endian integer
    /// which is a relative jump, based on the address following the reloc.
    rel32: usize,
    /// A branch in the ARM instruction set
    arm_branch: struct {
        pos: usize,
        cond: @import("codegen/arm.zig").Condition,
    },
};

pub const Result = union(enum) {
    /// The `code` parameter passed to `generateSymbol` has the value appended.
    appended: void,
    /// The value is available externally, `code` is unused.
    externally_managed: []const u8,
    fail: *ErrorMsg,
};

pub const GenerateSymbolError = error{
    OutOfMemory,
    /// A Decl that this symbol depends on had a semantic analysis failure.
    AnalysisFail,
};

pub const DebugInfoOutput = union(enum) {
    dwarf: struct {
        dbg_line: *std.ArrayList(u8),
        dbg_info: *std.ArrayList(u8),
        dbg_info_type_relocs: *link.File.DbgInfoTypeRelocsTable,
    },
    none,
};

pub fn generateSymbol(
    bin_file: *link.File,
    src_loc: Module.SrcLoc,
    typed_value: TypedValue,
    code: *std.ArrayList(u8),
    debug_output: DebugInfoOutput,
) GenerateSymbolError!Result {
    const tracy = trace(@src());
    defer tracy.end();

    switch (typed_value.ty.zigTypeTag()) {
        .Fn => {
            switch (bin_file.options.target.cpu.arch) {
                .wasm32 => unreachable, // has its own code path
                .wasm64 => unreachable, // has its own code path
                .arm => return ArmFunction(.arm).generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .armeb => return ArmFunction(.armeb).generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .aarch64 => return AArch64Function(.aarch64).generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .aarch64_be => return AArch64Function(.aarch64_be).generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .aarch64_32 => return AArch64Function(.aarch64_32).generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.arc => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.avr => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.bpfel => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.bpfeb => return <Function>.bpfeb).generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.hexagon => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.mips => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.mipsel => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.mips64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.mips64el => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.msp430 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.powerpc => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.powerpc64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.powerpc64le => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.r600 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.amdgcn => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.riscv32 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .riscv64 => return RV64Function.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.sparc => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.sparcv9 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.sparcel => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.s390x => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .spu_2 => return SPU_2Function.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.tce => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.tcele => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.thumb => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.thumbeb => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.i386 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                .x86_64 => return X8664Function.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.xcore => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.nvptx => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.nvptx64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.le32 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.le64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.amdil => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.amdil64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.hsail => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.hsail64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.spir => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.spir64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.kalimba => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.shave => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.lanai => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.renderscript32 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.renderscript64 => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                //.ve => return <Function>.generateSymbol(bin_file, src_loc, typed_value, code, debug_output),
                else => @panic("Backend architectures that don't have good support yet are commented out, to improve compilation performance. If you are interested in one of these other backends feel free to uncomment them. Eventually these will be completed, but stage1 is slow and a memory hog."),
            }
        },
        .Array => {
            // TODO populate .debug_info for the array
            if (typed_value.val.castTag(.bytes)) |payload| {
                if (typed_value.ty.sentinel()) |sentinel| {
                    try code.ensureCapacity(code.items.len + payload.data.len + 1);
                    code.appendSliceAssumeCapacity(payload.data);
                    const prev_len = code.items.len;
                    switch (try generateSymbol(bin_file, src_loc, .{
                        .ty = typed_value.ty.elemType(),
                        .val = sentinel,
                    }, code, debug_output)) {
                        .appended => return Result{ .appended = {} },
                        .externally_managed => |slice| {
                            code.appendSliceAssumeCapacity(slice);
                            return Result{ .appended = {} };
                        },
                        .fail => |em| return Result{ .fail = em },
                    }
                } else {
                    return Result{ .externally_managed = payload.data };
                }
            }
            return Result{
                .fail = try ErrorMsg.create(
                    bin_file.allocator,
                    src_loc,
                    "TODO implement generateSymbol for more kinds of arrays",
                    .{},
                ),
            };
        },
        .Pointer => {
            // TODO populate .debug_info for the pointer
            if (typed_value.val.castTag(.decl_ref)) |payload| {
                const decl = payload.data;
                if (decl.analysis != .complete) return error.AnalysisFail;
                // TODO handle the dependency of this symbol on the decl's vaddr.
                // If the decl changes vaddr, then this symbol needs to get regenerated.
                const vaddr = bin_file.getDeclVAddr(decl);
                const endian = bin_file.options.target.cpu.arch.endian();
                switch (bin_file.options.target.cpu.arch.ptrBitWidth()) {
                    16 => {
                        try code.resize(2);
                        mem.writeInt(u16, code.items[0..2], @intCast(u16, vaddr), endian);
                    },
                    32 => {
                        try code.resize(4);
                        mem.writeInt(u32, code.items[0..4], @intCast(u32, vaddr), endian);
                    },
                    64 => {
                        try code.resize(8);
                        mem.writeInt(u64, code.items[0..8], vaddr, endian);
                    },
                    else => unreachable,
                }
                return Result{ .appended = {} };
            }
            return Result{
                .fail = try ErrorMsg.create(
                    bin_file.allocator,
                    src_loc,
                    "TODO implement generateSymbol for pointer {}",
                    .{typed_value.val},
                ),
            };
        },
        .Int => {
            // TODO populate .debug_info for the integer
            const info = typed_value.ty.intInfo(bin_file.options.target);
            if (info.bits == 8 and info.signedness == .unsigned) {
                const x = typed_value.val.toUnsignedInt();
                try code.append(@intCast(u8, x));
                return Result{ .appended = {} };
            }
            return Result{
                .fail = try ErrorMsg.create(
                    bin_file.allocator,
                    src_loc,
                    "TODO implement generateSymbol for int type '{}'",
                    .{typed_value.ty},
                ),
            };
        },
        else => |t| {
            return Result{
                .fail = try ErrorMsg.create(
                    bin_file.allocator,
                    src_loc,
                    "TODO implement generateSymbol for type '{s}'",
                    .{@tagName(t)},
                ),
            };
        },
    }
}

pub const StackAllocation = struct {
    inst: *ir.Inst,
    /// TODO do we need size? should be determined by inst.ty.abiSize()
    size: u32,
};

pub fn Branch(comptime MCValue: type) type {
    return struct {
        inst_table: std.AutoArrayHashMapUnmanaged(*ir.Inst, MCValue) = .{},

        pub fn deinit(self: *@This(), gpa: *Allocator) void {
            self.inst_table.deinit(gpa);
            self.* = undefined;
        }
    };
}
