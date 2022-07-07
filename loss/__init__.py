from loss.criterion import DSNLoss, JointClsLoss

def get_loss(args):
    weight = None
    if args.class_weights is not None:
        weight = args.class_weights.cuda()
    if args.model == 'caanet':
        criterion = JointClsLoss(ignore_index=args.ignore_label, weight=weight,bins=args.bin_size, 
                        ohem=args.ohem, thresh=args.ohem_thres, min_kept=args.ohem_keep)
    else:
        criterion = DSNLoss(ignore_index=args.ignore_label, weight=weight, 
                        ohem=args.ohem, thresh=args.ohem_thres, min_kept=args.ohem_keep)
    return criterion
