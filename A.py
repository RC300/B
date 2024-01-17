import cv2, math, numpy, torch, argparse, torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-m', '--model')
parser.add_argument('-o', '--output')
args = parser.parse_args()

I0 = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
W0 = math.ceil(I0.shape[1] / I0.shape[0] * 512)
IV = I0.size // 120
C0 = numpy.dsplit(I0, 3)
SR = numpy.sort(C0[0], None)
SG = numpy.sort(C0[1], None)
SB = numpy.sort(C0[2], None)
V1 = numpy.clip(C0[0], SR[IV], SR[-IV])
V2 = numpy.clip(C0[1], SG[IV], SG[-IV])
V3 = numpy.clip(C0[2], SB[IV], SB[-IV])
R0 = cv2.normalize(V1, None, 0, 255, cv2.NORM_MINMAX)
G0 = cv2.normalize(V2, None, 0, 255, cv2.NORM_MINMAX)
B0 = cv2.normalize(V3, None, 0, 255, cv2.NORM_MINMAX)
NM = numpy.dstack((R0, G0, B0))

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(3, 64, 7), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.Conv2d(128, 256, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.Conv2d(256, 512, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 1024, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU())
        self.model += torch.nn.Sequential(ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock())
        self.model += torch.nn.Sequential(torch.nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU(),
        torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(64, 3, 7), torch.nn.Tanh())

    def forward(self, IN):
        IR = cv2.resize(IN, (W0, 512), interpolation=cv2.INTER_CUBIC)
        TS = torchvision.transforms.ToTensor()(IR)
        TR = self.model(TS) * 0.5 + 0.5
        TI = torchvision.transforms.functional.crop(TR, 0, 0, 512, W0)
        TI = torchvision.transforms.ConvertImageDtype(torch.uint8)(TI)
        NI = TI.numpy().transpose((1, 2, 0))
        return cv2.resize(NI, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_CUBIC)

class ResBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, 3), torch.nn.InstanceNorm2d(1),
        torch.nn.ReLU(), torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, 3), torch.nn.InstanceNorm2d(1))

    def forward(self, TS): return self.conv_block(TS) + TS

GT = Network()

GT.load_state_dict(torch.load(args.model + '1.lib'))
with torch.inference_mode(): NA = GT.forward(I0)
M1 = cv2.inRange(NA, (0, 255, 0), (0, 255, 0))
M2 = cv2.bitwise_not(M1)
N1 = cv2.bitwise_or(NA, NA, mask=M1) + cv2.bitwise_or(I0, I0, mask=M2)
cv2.imwrite(args.output + 'X1.jpg', cv2.cvtColor(N1, cv2.COLOR_BGR2RGB))

GT.load_state_dict(torch.load(args.model + '2.lib'))
with torch.inference_mode(): NB = GT.forward(NA)
P0 = cv2.inRange(NB, (255, 0, 0), (255, 0, 0))
N2 = cv2.bitwise_or(NB, NB, mask=M1) + cv2.bitwise_or(I0, I0, mask=M2)
for CT in cv2.findContours(P0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]:
    (x, y), r = cv2.minEnclosingCircle(CT)
    x, y, r = int(x), int(y), int(r)
    cv2.circle(N2, (x, y), r, (255, 0, 0), -1)
cv2.imwrite(args.output + 'X2.jpg', cv2.cvtColor(N2, cv2.COLOR_BGR2RGB))

P1 = cv2.inRange(N2, (0, 0, 0), (0, 0, 0))
P2 = cv2.inRange(N2, (255, 0, 0), (255, 0, 0))
P3 = cv2.inRange(N2, (255, 0, 255), (255, 0, 255))
P4 = cv2.inRange(N2, (0, 0, 255), (0, 0, 255))
for P in P1, P2, P3, P4:
    for CN in cv2.findContours(P, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]:
        (x, y), r = cv2.minEnclosingCircle(CN)
        x, y, r = math.ceil(x), math.ceil(y), math.ceil(r)
        if P is P1: cv2.circle(N1, (x, y), r, (0, 205, 0), -1)
        if P is P2: cv2.circle(N1, (x, y), r, (255, 0, 0), -1)
        if P is P2: cv2.circle(N1, (x, y), math.ceil(r / 4), (255, 255, 255), -1)
        if P is P3: cv2.circle(N1, (x, y), r, (255, 0, 255), -1)
        if P is P4: cv2.circle(N1, (x, y), r, (0, 0, 255), -1)
N3 = cv2.bitwise_or(N1, N1, mask=M1) + cv2.bitwise_or(NM, NM, mask=M2)
cv2.imwrite(args.output + 'X3.jpg', cv2.cvtColor(N3, cv2.COLOR_BGR2RGB))

GT.load_state_dict(torch.load(args.model + '3.lib'))
with torch.inference_mode(): ND = GT.forward(N3)
C4 = numpy.dsplit(ND, 3)
R4 = cv2.normalize(C4[0], None, SR[IV], SR[-IV], cv2.NORM_MINMAX)
G4 = cv2.normalize(C4[1], None, SG[IV], SG[-IV], cv2.NORM_MINMAX)
B4 = cv2.normalize(C4[2], None, SB[IV], SB[-IV], cv2.NORM_MINMAX)
N4 = numpy.dstack((R4, G4, B4))
N4 = cv2.bitwise_or(N4, N4, mask=M1) + cv2.bitwise_or(I0, I0, mask=M2)
cv2.imwrite(args.output + 'X4.jpg', cv2.cvtColor(N4, cv2.COLOR_BGR2RGB))