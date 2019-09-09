import cv2
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models # use torchvision >= 0.3.0
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np
from enum import Enum

from CNN_feat.MobileNetv3_feat import MobileNet_feat


os_env = 'MAC'
use_gpu = torch.cuda.is_available()
use_pth = False
feat_visual = False
cnn_type = 'x0_5'

if os_env == 'MAC':
	save_img = '/Users/apple/Downloads/cnnKCF_proj_code/KCF_8.12/CNN_feat/img_save/'
	path_img = '/Users/apple/Downloads/BenchMark/dataset_5/train/Soccer/'
	save_feat= '/Users/apple/Downloads/cnnKCF_proj_code/KCF_8.12/CNN_feat/feat_save/'
	path_pth = '/Users/apple/.cache/torch/checkpoints/'
elif os_env == 'WIN':
	save_img = 'img_save\\'
	path_img = 'David\\'
	save_feat = 'feat_save\\'
	path_pth = None

class LightCNN(Enum):
	SqueezeNet, MobileNet, ShuffleNet, MnasNet = 8,9,10,11


def draw_features(width, height, x, savename):
	tic=time.time()
	fig = plt.figure(figsize=(16, 16))
	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
	for i in range(width*height):
		plt.subplot(height,width, i + 1)
		plt.axis('off')
		# plt.tight_layout()
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

preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)) ])


def returnCAM(feature_conv, weight_softmax, class_idx):
	# generate the class activation maps upsample to 256x256
	size_upsample = (224, 224)
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


class LightWeight_feat(torch.nn.Module):
	def __init__(self, model_type=8, cnn_type='x0_5', cnn_layer_test=0):
		super(LightWeight_feat, self).__init__()
		self.model_type = model_type
		self.cnn_type = cnn_type
		self.use_layer = cnn_layer_test
		self.bn = True # BenchNorm
		
		# select cnn_model from SqueezeNet, MobileNet, ShuffleNet, MnasNet
		if self.model_type == LightCNN.SqueezeNet.value: # use SqueezeNet
			if cnn_type == '1_0': # V1_1_0
				self.model = models.squeezenet1_0(pretrained=True)
				self.sel_feat_layer = {0, 4, 8, 12} # conv, conv2_2, conv3_2, cam
			else: # V1_x1_1
				self.cnn_type = '1_1'
				self.model = models.squeezenet1_1(pretrained=True)
				self.sel_feat_layer = {0, 4, 7, 12} # conv, conv2_2, conv3_2, cam
			print('use SqueezeNet',self.cnn_type)
		
		elif self.model_type == LightCNN.MobileNet.value: # use MobileNet
			if cnn_type != 'Small':
				self.cnn_type = 'Large'
			self.model = MobileNet_feat(self.cnn_type,True,1000) # Large, Small
			
		elif self.model_type == LightCNN.ShuffleNet.value: # use ShuffleNet
			if cnn_type == 'x0_5': # V2_x0.5
				self.model = models.shufflenet_v2_x0_5(pretrained=True)
			else: # V2_x1.0
				self.cnn_type = 'x1_0'
				self.model = models.shufflenet_v2_x1_0(pretrained=True)
			print('use ShuffleV2',self.cnn_type)
		
		elif self.model_type == LightCNN.MnasNet.value: # use MnasNet
			if cnn_type == '0_5': # MnasNet_0_5
				self.model = models.mnasnet0_5(pretrained=True)
			else: # MnasNet_1_0
				self.cnn_type = '1_0'
				self.model = models.mnasnet1_0(pretrained=True)
			self.sel_feat_layer = {6, 8, 10, 16} # conv1_3, conv2, conv4, cam
			print('use MnasNet',self.cnn_type)
		
		self.model.eval()
		# using Grad-CAM with softmax weight (FC)
		params = list(self.model.parameters())
		self.weight_softmax = np.squeeze(params[-2].data.numpy())

		if use_gpu:
			print('use cuda model')
			self.model = self.model.cuda()
		if self.model_type == LightCNN.SqueezeNet.value:
			self.features = nn.ModuleList(list(self.model.features)).eval()
		elif self.model_type == LightCNN.MnasNet.value:
			self.features = nn.ModuleList(list(self.model.layers)).eval()
		
		# test feature
		if self.use_layer == 0: # run combine cnn_features
			self.use_layer = -4
			print('test linear combined cnn_features')
		elif self.use_layer > 4: # run combine features with grad-CAM
			print('test combined cnn_features with grad-CAM')
		else: # test each conv layers
			print('test single conv layer', self.use_layer)
		

	def forward(self, x):
		feat = []
		sel_layer = abs(self.use_layer)
		if self.model_type == LightCNN.SqueezeNet.value or self.model_type == LightCNN.MnasNet.value: # SqueezeNet MnasNet
			for ii, model in enumerate(self.features):
				x = model(x)
				if ii in self.sel_feat_layer: # get feat layer
					# print("get featture: ", ii, x.shape)
					feat.append(x.cpu().numpy())
					if (len(feat) == sel_layer):
						return x,feat
			# run classifier (GAP+FC)
			if self.model_type == LightCNN.SqueezeNet.value:
				x = self.model.classifier(x)
				x = torch.flatten(x, 1)
			else:
				x = x.mean([2, 3])
				x = self.model.classifier(x)
		
		elif self.model_type == LightCNN.ShuffleNet.value: # ShuffleNet
			x = self.model.conv1(x)
			feat.append(x.cpu().numpy()) # feat1 24*112*112
			if sel_layer == 1: return x,feat
			
			x = self.model.maxpool(x)
			x = self.model.stage2(x)
			feat.append(x.cpu().numpy()) # feat2 48*28*28
			if sel_layer == 2: return x,feat
			
			x = self.model.stage3(x)
			feat.append(x.cpu().numpy()) # feat3 96*14*14
			if sel_layer == 3: return x,feat
			
			x = self.model.stage4(x)
			x = self.model.conv5(x)
			feat.append(x.cpu().numpy()) # feat4 1024*7*7
			# use globalpool + FC
			x = x.mean([2, 3])
			x = self.model.fc(x)

		elif self.model_type == LightCNN.MobileNet.value: # MobileNet
			x = self.model.hs1(self.model.bn1(self.model.conv1(x))) # 16*112*112
			feat.append(x.cpu().numpy()) # feat1 112
			x = self.model.bneck(x)
			if sel_layer == 1: return x,feat
			
			x = self.model.conv2(x) # feat2 (cov2)
			feat.append(x.cpu().numpy()) # feat2 7
			if sel_layer == 2: return x,feat
			x = self.model.hs2(self.model.bn2(x)) # 960*7*7
			feat.append(x.cpu().numpy()) # feat2 7
			if sel_layer == 3: return x,feat

			x = F.avg_pool2d(x, 7)
			x = x.view(x.size(0), -1)
			x = self.model.hs3(self.model.bn3(self.model.linear3(x)))
			feat.append(x.cpu().numpy()) # feat3 CAM
			x = self.model.linear4(x)
		return x, feat

	def get_cnn_feat(self, img, feat_visual=False, use_CAM=False, save_cam=False):
		height, width = 224,224
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_tensor = preprocess(img)
		img_tensor = img_tensor.unsqueeze(0)
		if use_gpu:
			img_tensor = img_tensor.cuda()

		# use_CAM = False
		with torch.no_grad():
			out,feat = self.forward(img_tensor)
			if feat_visual and self.use_layer>2:
				print('feat shape:', feat[0].shape, feat[1].shape)
				if self.model_type == LightCNN.ShuffleNet.value:
					draw_features(4, 4, feat[0], save_feat + 'Shuffle'+self.cnn_type+'_feat_112.png')
					draw_features(6, 6, feat[1], save_feat + 'Shuffle'+self.cnn_type+'_feat_28.png')
				elif self.model_type == LightCNN.SqueezeNet.value:
					draw_features(8, 8, feat[0], save_feat + 'Squeeze'+self.cnn_type+'_feat_109.png')
					draw_features(10,10,feat[1], save_feat + 'Squeeze'+self.cnn_type+'_feat_54.png')
				elif self.model_type == LightCNN.MnasNet.value:
					draw_features(4, 4, feat[0], save_feat + 'MnasNet'+self.cnn_type+'_feat_109.png')
					draw_features(4, 4, feat[1], save_feat + 'MnasNet'+self.cnn_type+'_feat_56.png')
				elif self.model_type == LightCNN.MobileNet.value:
					draw_features(4, 4, feat[0], save_feat + 'MobileNet'+self.cnn_type+'_feat_112.png')
					draw_features(8, 8, feat[1], save_feat + 'MobileNet'+self.cnn_type+'_feat_7.png')
			
			# CAN'T use CAM with all cnn
			use_CAM = False
			if use_CAM:
				print('feat_cam: ', feat[2].shape, 'logit out: ', out.shape)
				# get class probs
				h_x = F.softmax(out, dim=1).data.squeeze()
				print('h_x: ', h_x.shape)
				probs, idx = h_x.sort(0, True)
				if use_gpu:
					probs,idx = probs.cpu(),idx.cpu()
				probs = probs.numpy()
				idx = idx.numpy()
				# run Grad_CAM with top 1 class
				CAMs = returnCAM(feat[2], self.weight_softmax, [idx[0]])
				heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
				if save_cam:
					img.save(save_img + 'Shuffle_test.jpg')
					img = cv2.imread(save_img + 'Shuffle_test.jpg')
					result = heatmap * 0.3 + img * 0.5
					cv2.imwrite(save_img + 'Shuffle_CAM.jpg', result)
					print('saved CAM_img')

			if self.use_layer <= 0: # linear combined cnn_features
				# use conv layer 1,3,4
				feat.pop(1)
				return feat
			elif self.use_layer > 4:
				# use conv layer 1,2,3 + grad-CAM
				print('Cannot use CAM of lightweight cnn!')
				# feat.append(self.CAMs)
				return feat
			else: # test single feat layer
				# print(feat[0].shape, feat[0].dtype)
				return feat[self.use_layer-1]


