import PIL, cv2, math, numpy, torch, argparse, torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-m', '--model')
parser.add_argument('-o', '--output')
args = parser.parse_args()

I0 = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
W0 = math.ceil(I0.shape[1] / I0.shape[0] * 512)
RI = cv2.resize(I0, (W0, 512), interpolation=cv2.INTER_CUBIC)
IV = RI.size // 120
C0 = numpy.dsplit(RI, 3)
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

class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(3, 64, 7), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.Conv2d(128, 256, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.Conv2d(256, 512, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.Conv2d(512, 1024, 3, 2, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [ResnetBlock() for i in range(9)]
        model += [torch.nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        model += [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(64, 3, 7), torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, tensor): return self.model(tensor)

class ResnetBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        conv_block = [torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, 3), torch.nn.InstanceNorm2d(1), torch.nn.ReLU()]
        conv_block += [torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, 3), torch.nn.InstanceNorm2d(1)]
        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, tensor): return tensor + self.conv_block(tensor)

GE = Generator()

T1 = torchvision.transforms.ToTensor()(RI).unsqueeze(0)
L1 = os.path.join(args.model, '1.lib')
GE.load_state_dict(torch.load(L1))
with torch.inference_mode(): T1 = GE(T1)
T1 = T1.squeeze() * 0.5 + 0.5
T1 = torchvision.transforms.functional.convert_image_dtype(T1, torch.uint8)
IA = torchvision.transforms.ToPILImage()(T1)
IA = IA.crop((0, 0, W0, 512))
NA = numpy.array(IA)
M1 = cv2.inRange(NA, (0, 255, 0), (0, 255, 0))
N1 = cv2.bitwise_or(RI, RI, mask=cv2.bitwise_not(M1)) + cv2.bitwise_or(NA, NA, mask=M1)
I1 = cv2.resize(N1, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_CUBIC)
M2 = cv2.inRange(I1, (0, 255, 0), (0, 255, 0))
cv2.imwrite(os.path.join(args.output, 'X1.jpg'), cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

T2 = torchvision.transforms.ToTensor()(NA).unsqueeze(0)
L2 = os.path.join(args.model, '2.lib')
GE.load_state_dict(torch.load(L2))
with torch.inference_mode(): T2 = GE(T2)
T2 = T2.squeeze() * 0.5 + 0.5
T2 = torchvision.transforms.functional.convert_image_dtype(T2, torch.uint8)
IB = torchvision.transforms.ToPILImage()(T2)
IB = IB.crop((0, 0, W0, 512))
NB = numpy.array(IB)
N2 = cv2.bitwise_or(RI, RI, mask=cv2.bitwise_not(M1)) + cv2.bitwise_or(NB, NB, mask=M1)
I2 = cv2.resize(N2, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(args.output, 'X2.jpg'), cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

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
N3 = cv2.bitwise_or(NM, NM, mask=cv2.bitwise_not(M1)) + cv2.bitwise_or(N1, N1, mask=M1)
I3 = cv2.resize(N3, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(args.output, 'X3.jpg'), cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))

T4 = torchvision.transforms.ToTensor()(N3).unsqueeze(0)
L3 = os.path.join(args.model, '3.lib')
GE.load_state_dict(torch.load(L3))
with torch.inference_mode(): T4 = GE(T4)
T4 = T4.squeeze() * 0.5 + 0.5
ID = torchvision.transforms.ToPILImage()(T4)
ID = ID.crop((0, 0, W0, 512))
DN = numpy.array(ID)
C4 = numpy.dsplit(DN, 3)
R4 = cv2.normalize(C4[0], None, SR[IV], SR[-IV], cv2.NORM_MINMAX)
G4 = cv2.normalize(C4[1], None, SG[IV], SG[-IV], cv2.NORM_MINMAX)
B4 = cv2.normalize(C4[2], None, SB[IV], SB[-IV], cv2.NORM_MINMAX)
N4 = numpy.dstack((R4, G4, B4))
DR = cv2.resize(N4, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_CUBIC)
I4 = cv2.bitwise_or(I0, I0, mask=cv2.bitwise_not(M2)) + cv2.bitwise_or(DR, DR, mask=M2)
cv2.imwrite(os.path.join(args.output, 'X4.jpg'), cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))