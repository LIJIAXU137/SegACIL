import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade','cityscapes_domain'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50,
                        help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
                        help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument('--unknown', type=int, default=0, help='unknown 属性的默认值')


    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--gpu_id", type=str, default='1',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    
    parser.add_argument("--subpath", type=str, default=None, help="subpath for saving ckpt and log")
    
    # CIL options
    parser.add_argument("--pseudo", action='store_true', help="enable pseudo-labeling")
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task", type=str, default='15-1', help="cil task")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', help="overlap setup (True), disjoint setup (False)")
    parser.add_argument("--setting", type=str, default='overlap', choices=['sequential', 'disjoint', 'overlap'],
                        help="continual learning setting: 'sequential', 'disjoint', 'overlap'")
    parser.add_argument("--bn_freeze", action='store_true', help="enable batchnorm freezing")
    parser.add_argument("--cil_step", type=int, default=0, help="cil step")
    parser.add_argument("--initial", action='store_true', help="initial training")

    parser.add_argument("--buffer", type=int, default=16384, help="buffer size")
    
    parser.add_argument("--gamma", type=float, default=10, help="gamma value")
    parser.add_argument("--method", type=str, default='None', help="method value")
    # parser.add_argument("--mem_size", type=int, default=10, help="memory size")

    return parser.parse_args()