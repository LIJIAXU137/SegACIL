import numpy as np
from sklearn.metrics import confusion_matrix

VOC_CLASSES = [
        "background", "aeroplane", "bicycle", "bird",
        "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

ADE_CLASSES = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]
CITYSCAPES_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                      "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                      "bicycle"]

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, num_classes, dataset):
        self.num_classes = num_classes
        self.n_classes = sum(num_classes)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
        if dataset == 'voc':
            self.CLASSES = VOC_CLASSES
        elif dataset == 'ade':
            self.CLASSES = ADE_CLASSES
        else:
            self.CLASSES = CITYSCAPES_CLASSES
        
    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )

    def to_str_val(self, results):
        string = "\n"
        
        string+='Class IoU/Acc:\n'
        for (k, v1), v2 in zip(results['Class IoU'].items(), results['Class Acc'].values()):
            if k==0 or k>=self.n_classes-self.num_classes[-1]:
                # print((f"Index {k} is out of range for CLASSES with length {len(self.CLASSES)}"))
                string += "\%s: %.4f (miou) , %.4f (acc) \n" % (self.CLASSES[k], v1, v2)
        return string
    
    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU" and k!="Class Acc":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU/Acc:\n'
        for (k, v1), v2 in zip(results['Class IoU'].items(), results['Class Acc'].values()):
            string += "\%s: %.4f (miou) , %.4f (acc) \n" % (self.CLASSES[k], v1, v2)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + EPS)
        cls_acc = dict(zip(range(self.n_classes), acc_cls))
        acc_cls = np.nanmean(acc_cls)
        
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + EPS)
        #mean_iu = np.nanmean(iu[iu.nonzero()])
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Class Acc": cls_acc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]




class StreamSegMetrics2(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, num_classes):
       
        self.n_classes = sum(num_classes)
        
        self.confusion_matrix = np.zeros(( self.n_classes ,  self.n_classes ))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str_val(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Acc" and k != "Confusion Matrix":
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string
    
    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Acc" and k != "Confusion Matrix":
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(
            zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)])
        )

        return {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,

        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0


