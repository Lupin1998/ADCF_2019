import cv2
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np

os_env = 'WIN'
use_gpu = torch.cuda.is_available()
feat_visual = True
cnn_type = 50

if os_env == 'MAC':
	save_img = '/Users/apple/Downloads/CNN_feat/img_save/'
	path_img = '/Users/apple/Downloads/BenchMark/dataset_5/train/Soccer/'
	save_feat= '/Users/apple/Downloads/CNN_feat/feat_save/'
	path_pth = '/Users/apple/.torch/models/seresnet50-60a8950a85b2b.pkl.mdlp' # use SEResNet model
elif os_env == 'WIN':
	save_img = 'img_save\\'
	path_img = 'David\\'
	save_feat = 'feat_save\\'
	path_pth = 'seresnet50-60a8950a85b2b.pkl.mdlp'

preprocess = transforms.Compose([transforms.Resize((224,224)),
					transforms.ToTensor(),
					transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

def draw_features(width, height, x, savename):
	tic=time.time()
	fig = plt.figure(figsize=(16, 16))
	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
	for i in range(width*height):
		plt.subplot(height,width, i + 1)
		plt.axis('off')
		# plt.tight_layout()
		#print('i:', i)
		img = x[0, i, :, :]
		pmin = np.min(img)
		pmax = np.max(img)
		img = (img - pmin) / (pmax - pmin + 0.000001)
		plt.imshow(img, cmap='gray')
		print("{}/{}".format(i,width*height))
	fig.savefig(savename, dpi=100)
	fig.clf()
	plt.close()
	print("time:{}".format(time.time()-tic))

def returnCAM(feature_conv, weight_softmax, class_idx):
	# generate the class activation maps upsample to 256x256
	size_upsample = (256, 256)
	bz, nc, h, w = feature_conv.shape
	print('feature_conv.shape: ', feature_conv.shape)
	output_cam = []
	for idx in class_idx:
		print('weight_softmax[idx]: ', weight_softmax[idx].shape, weight_softmax[idx])
		cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
		cam = cam.reshape(h, w)
		cam = cam - np.min(cam)
		cam_img = cam / np.max(cam) # get CAM feature
		print('CAM feature: ', cam_img.shape)
		cam_img = np.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample))
	return output_cam


from senet.se_resnet import se_resnet50

class SE_Resnet_feat(nn.Module):
	def __init__(self):
		super(SE_Resnet_feat, self).__init__()
		self.model = se_resnet50()
		load = torch.load(path_pth)
		self.model.load_state_dict(load)
		print('SEResNet50 models loaded!')
		
		# using Grad-CAM with softmax weight
		params = list(self.model.parameters())
		self.weight_softmax = np.squeeze(params[-2].data.numpy())
		
		self.test = True
		self.model.eval()
		if use_gpu:
			self.model = self.model.cuda()

	def forward(self, x):
		print('x input: ', x.shape)
		x = self.model.conv1(x)
		feat = x.detach().numpy()
		# print('feat: ', feat.shape)
		
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)
		feat_cam = x.detach().numpy()
		# print('feat cam: ', feat_cam.shape)
		
		x = self.model.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.model.fc(x)
		return x, feat, feat_cam
	
	
	def get_cnn_feat(self, img, feat_visual=False, use_CAM=False, save_cam=False):
		height, width = 224,224
		if img.format==None and feat_visual:
			img = Image.open(path_img + '000015.jpg')
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_tensor = preprocess(img)
		img_tensor = img_tensor.unsqueeze(0)
		if use_gpu:
			img_tensor = img_tensor.cuda()
		
		out,feat,feat_cam = self.forward(img_tensor)
		if feat_visual:
			print('feat shape:', feat.shape)
			draw_features(8,8, feat, save_feat + "SE_Resnet_112.png")
		if use_CAM:
			print('feat_cam: ', feat_cam.shape, 'logit out: ', out.shape)
			# get class probs
			h_x = F.softmax(out, dim=1).data.squeeze()
			print('h_x: ', h_x.shape)
			probs, idx = h_x.sort(0, True)
			probs = probs.numpy()
			idx = idx.numpy()
			# run Grad_CAM with top 1 class
			CAMs = returnCAM(feat_cam, self.weight_softmax, [idx[0]])
			heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
			if save_cam:
				img.save(save_img + 'SENet_test.jpg')
				img = cv2.imread(save_img + 'SENet_test.jpg')
				result = heatmap * 0.3 + img * 0.5
				cv2.imwrite(save_img + 'SENet_CAM.jpg', result)
				print('save SENet CAM')
		return feat

