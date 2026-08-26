import cv2, math, numpy, torch, argparse, torchvision.transforms.v2

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-m', '--model')
parser.add_argument('-o', '--output')
args = parser.parse_args()

I0 = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
W1 = math.ceil(I0.shape[1] / I0.shape[0] * 512)
IV = I0.size // 30
C0 = cv2.split(I0)
SR = numpy.sort(C0[0], None)
SG = numpy.sort(C0[1], None)
SB = numpy.sort(C0[2], None)
V1 = numpy.clip(C0[0], SR[IV], SR[-IV])
V2 = numpy.clip(C0[1], SG[IV], SG[-IV])
V3 = numpy.clip(C0[2], SB[IV], SB[-IV])
R0 = cv2.normalize(V1, None, 0, 255, cv2.NORM_MINMAX)
G0 = cv2.normalize(V2, None, 0, 255, cv2.NORM_MINMAX)
B0 = cv2.normalize(V3, None, 0, 255, cv2.NORM_MINMAX)
IS = cv2.merge((R0, G0, B0))

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(3, 64, 7), torch.nn.InstanceNorm2d(64), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.InstanceNorm2d(128), torch.nn.ReLU(),
        torch.nn.Conv2d(128, 256, 3, 2, 1), torch.nn.InstanceNorm2d(256), torch.nn.ReLU(),
        torch.nn.Conv2d(256, 512, 3, 2, 1), torch.nn.InstanceNorm2d(512), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 1024, 3, 2, 1), torch.nn.InstanceNorm2d(1024), torch.nn.ReLU())
        self.model += torch.nn.Sequential(ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock(), ResBlock())
        self.model += torch.nn.Sequential(torch.nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1), torch.nn.InstanceNorm2d(512), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), torch.nn.InstanceNorm2d(256), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), torch.nn.InstanceNorm2d(128), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), torch.nn.InstanceNorm2d(64), torch.nn.ReLU(),
        torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(64, 3, 7), torch.nn.Tanh())

    def __call__(self, IN):
        NI = cv2.resize(IN, (W1, 512), interpolation=cv2.INTER_CUBIC)
        TS = torchvision.transforms.v2.ToImage()(NI)
        TS = torchvision.transforms.v2.ToDtype(torch.float32, True)(TS)
        TR = self.model(TS) * 0.5 + 0.5
        TI = torchvision.transforms.v2.ToDtype(torch.uint8, True)(TR)
        NT = TI.permute(1, 2, 0).numpy()
        NC = NT[0:NI.shape[0], 0:NI.shape[1]]
        return cv2.resize(NC, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_CUBIC)

class ResBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, 3), torch.nn.InstanceNorm2d(1024),
        torch.nn.ReLU(), torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, 3), torch.nn.InstanceNorm2d(1024))

    def __call__(self, TS): return self.conv_block(TS) + TS

GT = Network()

GT.load_state_dict(torch.load(args.model + '1.lib'))
with torch.inference_mode(): IA = GT(I0)
M1 = cv2.inRange(IA, (0, 255, 0), (0, 255, 0))
M2 = cv2.bitwise_not(M1)
I1 = cv2.bitwise_or(IA, IA, mask=M1) + cv2.bitwise_or(I0, I0, mask=M2)
cv2.imwrite(args.output + 'X1.jpg', cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

GT.load_state_dict(torch.load(args.model + '2.lib'))
with torch.inference_mode(): IB = GT(IA)
I2 = cv2.bitwise_or(IB, IB, mask=M1) + cv2.bitwise_or(I0, I0, mask=M2)
cv2.imwrite(args.output + 'X2.jpg', cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

P1 = cv2.inRange(I2, (0, 0, 0), (255, 0, 0))
P2 = cv2.inRange(I2, (255, 0, 0), (255, 0, 0))
P3 = cv2.inRange(I2, (255, 0, 255), (255, 0, 255))
P4 = cv2.inRange(I2, (0, 0, 255), (0, 0, 255))
CT = cv2.findContours(P1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
CA = cv2.findContours(P2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
CB = cv2.findContours(P3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
CV = cv2.findContours(P4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
LT = sorted(a for a in (cv2.minEnclosingCircle(TC) for TC in CT) if a[1] > 30)
LA = sorted(b for b in (cv2.minEnclosingCircle(AC) for AC in CA) if b[1] > 10)
LB = sorted(c for c in (cv2.minEnclosingCircle(BC) for BC in CB) if c[1] > 10)
LV = sorted(d for d in (cv2.minEnclosingCircle(VC) for VC in CV) if d[1] > 10)
RA = max(e[1] for e in LA)
RT = (math.ceil(LT[1][0][0]) - math.ceil(LT[0][0][0])) / 2
RY = math.ceil(max(f[0][1] for f in LT) + RT) - (min(g[0][1] for g in LT) - RT)
print(math.ceil(RT))
print(math.ceil(RY))
cv2.circle(I1, (math.ceil(LT[0][0][0]), math.ceil(LT[0][0][1])), math.ceil(RT), (0, 205, 0), -1)
cv2.circle(I1, (math.ceil(LA[0][0][0]), math.ceil(LT[0][0][1])), math.ceil(RA), (255, 0, 0), -1)
cv2.circle(I1, (math.ceil(LA[0][0][0]), math.ceil(LT[0][0][1])), math.ceil(RA / 4), (255, 255, 255), -1)
cv2.circle(I1, (math.ceil(LT[1][0][0]), math.ceil(LT[1][0][1])), math.ceil(RT), (0, 205, 0), -1)
cv2.circle(I1, (math.ceil(LA[1][0][0]), math.ceil(LT[1][0][1])), math.ceil(RA), (255, 0, 0), -1)
cv2.circle(I1, (math.ceil(LA[1][0][0]), math.ceil(LT[1][0][1])), math.ceil(RA / 4), (255, 255, 255), -1)
if LB: cv2.circle(I1, (math.ceil(LB[0][0][0]), math.ceil(LB[0][0][1])), math.ceil(LB[0][1]), (255, 0, 255), -1)
if LV: cv2.circle(I1, (math.ceil(LV[0][0][0]), math.ceil(LV[0][0][1])), math.ceil(LV[0][1]), (0, 0, 255), -1)
I3 = cv2.bitwise_or(I1, I1, mask=M1) + cv2.bitwise_or(IS, IS, mask=M2)
cv2.imwrite(args.output + 'X3.jpg', cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))

GT.load_state_dict(torch.load(args.model + '3.lib'))
with torch.inference_mode(): ID = GT(I3)
C4 = cv2.split(ID)
R4 = cv2.normalize(C4[0], None, SR[IV], SR[-IV], cv2.NORM_MINMAX)
G4 = cv2.normalize(C4[1], None, SG[IV], SG[-IV], cv2.NORM_MINMAX)
B4 = cv2.normalize(C4[2], None, SB[IV], SB[-IV], cv2.NORM_MINMAX)
I4 = cv2.merge((R4, G4, B4))
I4 = cv2.bitwise_or(I4, I4, mask=M1) + cv2.bitwise_or(I0, I0, mask=M2)
cv2.imwrite(args.output + 'X4.jpg', cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))