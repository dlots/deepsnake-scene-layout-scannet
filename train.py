import argparse
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam as AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torchvision

mock_ds = torch.empty(3,32,32)

def train(data_loader):
    net = Linear(3,1)
    lr = 1e-3
    optimizer = AdamW(net.parameters(),lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 1, T_mult = 2)
    max_epochs = 40
    for ep_id in range(max_epochs):
        net.train()
        for b_id, batch in enumerate(data_loader):
            optimizer.zero_grad()
            output = net(batch['data'].to(data_loader))
            CEloss = CrossEntropyLoss()
            loss = CEloss(output,batch['ground_truth'])
            writer = SummaryWriter(log_dir = 'C:\Users\andre\OneDrive\Рабочий стол\train\checkpoints',
                comment= "Batch loss")
            writer.add_graph(loss)
            loss.backward()
            optimiser.step()
        scheduler.step()
        torch.save({
            'epoch':ep_id,
            'model_state_dict':net.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss,
        },'/checkpoints/checkpoints.txt')

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
if __name__ == "__main__":
    train(mock_ds)