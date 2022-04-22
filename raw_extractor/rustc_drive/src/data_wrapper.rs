use rustc_middle::{
    mir::{
        interpret::{AllocRange, ConstValue},
        BasicBlockData, ConstantKind, Operand, Rvalue, StatementKind,
    },
    ty::{self, TyCtxt},
};
use rustc_target::abi::Size;
use rustc_middle::ty::query::query_stored::promoted_mir;

use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize)]
pub struct MirBasicBlock {
    statements: Vec<MirStatement>,
    term: MirTerminator,
    is_cleanup: bool,
    ref_strs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MirTerminator {
    Goto {
        target: u32,
    },
    SwitchInt {
        targets: Vec<u32>,
    },
    Resume,
    Abort,
    Return,
    Unreachable,
    Drop {
        target: u32,
        unwind: Option<u32>,
    },
    DropAndReplace {
        target: u32,
        unwind: Option<u32>,
    },
    Call {
        func: String,
        args: Vec<String>,
        dest: Option<u32>,
        cleanup: Option<u32>,
    },
    Assert {
        cond: String,
        target: u32,
        cleanup: Option<u32>,
    },
    Yield {
        val: String,
        resume: u32,
        drop: Option<u32>,
    },
    GeneratorDrop,
    FalseEdge {
        real_target: u32,
        imaginary_target: u32,
    },
    FalseUnwind {
        real_target: u32,
        unwind: Option<u32>,
    },
    InlineAsm {
        dest: Option<u32>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MirStatement(String);

impl MirBasicBlock {
    pub fn new(statements: Vec<MirStatement>, term: MirTerminator, is_cleanup: bool, ref_strs: Vec<String>) -> Self {
        MirBasicBlock {
            statements,
            term,
            is_cleanup,
            ref_strs,
        }
    }
}

impl<'tcx> From<&rustc_middle::mir::StatementKind<'tcx>> for MirStatement {
    fn from(k: &rustc_middle::mir::StatementKind<'tcx>) -> Self {
        let expr = match k {
            StatementKind::Assign(b) => format!("assign {:?} = {:?}", b.0, b.1),
            StatementKind::FakeRead(b) => format!("fake {:?}", b.1),
            StatementKind::StorageLive(l) => format!("sl {:?}", l),
            StatementKind::StorageDead(l) => format!("sd {:?}", l),
            StatementKind::SetDiscriminant{place, variant_index, ..} => format!("set {:?} {:?}", place, variant_index),
            _ => format!("{:?}", k),
        };
        MirStatement(expr)
    }
}

impl<'tcx> From<&rustc_middle::mir::terminator::TerminatorKind<'tcx>> for MirTerminator {
    fn from(k: &rustc_middle::mir::terminator::TerminatorKind<'tcx>) -> Self {
        use rustc_middle::mir::terminator::TerminatorKind;
        match k {
            TerminatorKind::Goto { target } => Self::Goto {
                target: target.as_u32(),
            },
            TerminatorKind::SwitchInt { targets, .. } => Self::SwitchInt {
                targets: targets.all_targets().iter().map(|x| x.as_u32()).collect(),
            },
            TerminatorKind::Resume => Self::Resume,
            TerminatorKind::Abort => Self::Abort,
            TerminatorKind::Return => Self::Return,
            TerminatorKind::Unreachable => Self::Unreachable,
            TerminatorKind::Drop { target, unwind, .. } => Self::Drop {
                target: target.as_u32(),
                unwind: unwind.map(|x| x.as_u32()),
            },
            TerminatorKind::DropAndReplace { target, unwind, .. } => Self::DropAndReplace {
                target: target.as_u32(),
                unwind: unwind.map(|x| x.as_u32()),
            },
            TerminatorKind::Call {
                func,
                args,
                destination,
                cleanup,
                ..
            } => {
                let func = format!("{:?}", func);
                let args = args.iter().map(|x| format!("{:?}", x)).collect();
                let dest = destination.map(|(_, x)| x.as_u32());
                let cleanup = cleanup.map(|x| x.as_u32());

                Self::Call {
                    func,
                    args,
                    dest,
                    cleanup,
                }
            }
            TerminatorKind::Assert {
                cond, target, cleanup, ..
            } => Self::Assert {
                cond: format!("{:?}", cond),
                target: target.as_u32(),
                cleanup: cleanup.map(|x| x.as_u32()),
            },
            TerminatorKind::Yield { value, resume, drop, .. } => Self::Yield {
                val: format!("{:?}", value), 
                resume: resume.as_u32(),
                drop: drop.map(|x| x.as_u32()),
            },
            TerminatorKind::GeneratorDrop => Self::GeneratorDrop,
            TerminatorKind::FalseEdge {
                real_target,
                imaginary_target,
            } => Self::FalseEdge {
                real_target: real_target.as_u32(),
                imaginary_target: imaginary_target.as_u32(),
            },
            TerminatorKind::FalseUnwind {
                real_target,
                unwind,
            } => Self::FalseUnwind {
                real_target: real_target.as_u32(),
                unwind: unwind.map(|x| x.as_u32()),
            },
            TerminatorKind::InlineAsm { destination, .. } => Self::InlineAsm {
                dest: destination.map(|x| x.as_u32()),
            },
        }
    }
}

fn str_const_from_operand<'tcx>(tyctxt: TyCtxt<'tcx>, opr: &Operand<'tcx>, prom: &promoted_mir<'tcx>) -> Option<String> {
    match opr {
        Operand::Constant(c) => match c.literal {
            // String literals, like
            // ~~~
            // let a = "Some string.";
            // ~~~
            ConstantKind::Val(_val, _ty) => {
                if let ty::Ref(_, ty, _) = _ty.kind() {
                    if let ty::Str = ty.kind() {
                        // Slice, used only for &[u8] and &str
                        if let ConstValue::Slice{ data, start, end } = _val {
                            let data = data.0
                                .get_bytes(
                                    &tyctxt,
                                    AllocRange {
                                        start: Size::from_bytes(start),
                                        size: Size::from_bytes(end - start),
                                    },
                                )
                                .unwrap();
                            let s = String::from_utf8_lossy(data).to_string();
                            // println!("data = {}", s);
                            return Some(s);
                        }
                    }
                }
                None
            }

            // Formatted strings, like
            // ~~~
            // let a = format!("{} Test {} String", 4, 5);
            // ~~~
            ConstantKind::Ty(cst) => {


                if let rustc_middle::ty::ConstKind::Unevaluated(uneval) = cst.val() {
                    if let Some(promoted) = uneval.promoted {
                        if let Some(promoted_body) = prom.get(promoted) {
                            let str_vec = promoted_body
                                .basic_blocks()
                                .iter()
                                .map(|bb|  {
                                    // TODO: ChangeMe to use vectors rather than Option<String>
                                    // In current design, each opr can return at most one string.
                                    //   However, 
                                    //     1) If Rvalue is of type aggregate, it may corresponds
                                    //       to multiple strings.
                                    //     2) If ConstKind is of type Unevaluated, the promoted case
                                    //       it corresponds to another mir body, which may contain
                                    //       multiple bbs thus multiple strings.
                                    get_bb_refed_strs(tyctxt, &bb, prom).join("")
                                })
                                .collect::<Vec<_>>();
                            if str_vec.len() > 0 {
                                return Some(str_vec.join(""))
                            }
                            
                        }
                    }
                }

                match cst.ty().kind() {
                    // The code below may work in a stale version
                    //     if let TyKind::Str = ty.kind() {
                    //         if let ConstKind::Value(val) = cst.val() {
                    //             if let ConstValue::Slice { data, start, end } = val {
                    //                 let data = data.0
                    //                     .get_bytes(
                    //                         &tyctxt,
                    //                         AllocRange {
                    //                             start: Size::from_bytes(start),
                    //                             size: Size::from_bytes(end - start),
                    //                         },
                    //                     )
                    //                     .unwrap();
                    //                 let s = String::from_utf8_lossy(data).to_string();
                    //                 // println!("data = {}", s);
                    //                 return Some(s);
                    //             }
                    //         } // str
                    //     }
                    //     return None;
                    // },
                    // TyKind::Int(_) => {
                    //     println!("Gotcha! Int");
                    //     None
                    // },
                    _ => None
                }
            }
        }
        _ => None,
    }
}

pub fn get_bb_refed_strs<'tcx>(tyctxt: TyCtxt<'tcx>, bb: &BasicBlockData<'tcx>, prom: &promoted_mir<'tcx>) -> Vec<String> {
    // strs from statements
    let mut ref_strs: Vec<String> = bb.statements
        .iter()
        .filter_map(|stmt| match &stmt.kind {
            StatementKind::Assign(b) => match &b.1 {
                Rvalue::Use(opr) => str_const_from_operand(tyctxt, &opr, prom),
                Rvalue::Repeat(opr, _) => str_const_from_operand(tyctxt, opr, prom),
                Rvalue::Cast(_, opr, _) => str_const_from_operand(tyctxt, opr, prom),
                Rvalue::BinaryOp(_, ops) => str_const_from_operand(tyctxt, &ops.0, prom),
                Rvalue::Aggregate(_, v) => {
                    let str_vec = v.iter().filter_map(|opr| str_const_from_operand(tyctxt, opr, prom)).collect::<Vec<_>>();
                    if str_vec.len() > 0 {
                        Some(str_vec.join(""))
                    } else {
                        None
                    }
                }
                _ => None,

            },
            _ => None,
        })
        .collect();
    // It is also possible to ref strs in function arguments
    if let rustc_middle::mir::terminator::TerminatorKind::Call{args, ..} = &(bb.terminator().kind) {
        let mut args_strs = args.iter().filter_map(|opr| str_const_from_operand(tyctxt, opr, prom)).collect::<Vec<_>>();
        ref_strs.append(&mut args_strs);
    }
    ref_strs
}