"""Generate verify.py-compatible predictions from the trained SOTA LeNet model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda"
NORM = {'Ne': (5.31, 0.5), 'Nmu': (4.3, 0.42), 'Age': (1.0, 1.0), 'Ze': (0.0, 60.0)}

class LeNetArchitecture(nn.Module):
    def __init__(self, n_classes=5, n_reco_features=4, n_channels=2):
        super().__init__()
        ch = 16
        self.conv0 = nn.Conv2d(n_channels, ch, 2)
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.conv2 = nn.Conv2d(ch, ch, 4)
        self.conv3 = nn.Conv2d(ch, ch, 2)
        self.fc1 = nn.Linear(ch * 3 * 3 + n_reco_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.fc_on_conv = nn.Linear(ch * 3 * 3, n_classes)

    def forward(self, x, x_reco=None, return_cnn_out=False):
        x = x.permute(0, 3, 1, 2).float()
        x = F.max_pool2d(F.relu(self.conv0(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        if x_reco is not None:
            x = torch.cat((x, x_reco), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def main():
    # Load model
    model = LeNetArchitecture(n_classes=5, n_reco_features=4, n_channels=2).to(DEVICE)
    model.load_state_dict(torch.load("reproduce_sota_best.pt", weights_only=True))
    model.eval()

    # Load test data (from the new pipeline — exact match with published 30% split)
    matrices = np.load("data/composition_test/matrices.npy", mmap_mode="r")
    features = np.load("data/composition_test/features.npy", mmap_mode="r")
    n = len(matrices)
    print(f"Test set: {n} events")

    # Features layout: [E, Ze, Az, Ne, Nmu, Age]
    # Model expects reco features: [Ne, Nmu, Age, Ze] normalized
    all_preds = []
    BS = 2048
    with torch.no_grad():
        for i in range(0, n, BS):
            mat = torch.from_numpy(np.array(matrices[i:i+BS]).astype(np.float32)).to(DEVICE)
            feat = np.array(features[i:i+BS]).astype(np.float32)
            Ne = (feat[:, 3] - NORM['Ne'][0]) / NORM['Ne'][1]
            Nmu = (feat[:, 4] - NORM['Nmu'][0]) / NORM['Nmu'][1]
            Age = feat[:, 5] - NORM['Age'][0]
            Ze = (feat[:, 1] - NORM['Ze'][0]) / NORM['Ze'][1]
            reco = torch.from_numpy(np.column_stack([Ne, Nmu, Age, Ze]).astype(np.float32)).to(DEVICE)
            out = model(mat, reco)
            all_preds.append(out.argmax(1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    np.savez("predictions_sota_baseline.npz", predictions=predictions)
    print(f"Saved predictions_sota_baseline.npz ({len(predictions)} predictions)")

if __name__ == "__main__":
    main()
