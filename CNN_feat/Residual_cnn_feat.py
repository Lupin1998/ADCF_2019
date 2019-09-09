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
from enum import Enum

os_env = 'MAC'
use_gpu = torch.cuda.is_available()
use_pth = False
feat_visual = True
cnn_type = 50

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

class ResCNN(Enum):
	ResNet, DenseNet, WideResNet, ResNext = 4,5,6,7

preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
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
	size_upsample = (224, 224)
	bz, nc, h, w = feature_conv.shape
	print('feature_conv.shape: ', feature_conv.shape)
	output_cam = []
	for idx in class_idx:
		print('weight_softmax[idx]: ', weight_softmax[idx].shape, weight_softmax[idx])
		cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
		cam = cam.reshape(h, w)
		print('raw cam shape:',cam.shape)
		cam = cam - np.min(cam)
		cam_img = cam / np.max(cam) # get CAM feature
		print('CAM feature: ', cam_img.shape)
		cam_img = np.uint8(255 * cam_img)
		print('cam_img shape:',cam_img.shape)
		output_cam.append(cv2.resize(cam_img, size_upsample)) # 224,224
	return output_cam


class Residual_feat(nn.Module):
	def __init__(self, model_type=4, cnn_type=50, cnn_layer_test=0):
		super(Residual_feat, self).__init__()
		self.model_type = model_type
		self.cnn_type = cnn_type
		self.use_layer = cnn_layer_test
		
		# select cnn_model from ResNet, DenseNet, WideResNet, ResNext
		if self.model_type == ResCNN.ResNet.value: # use ResNet
			if cnn_type == 34:
				self.model = models.resnet34(pretrained=True)
			elif cnn_type == 101:
				self.model = models.resnet101(pretrained=True)
			else:
				self.cnn_type = 50
				self.model = models.resnet50(pretrained=True)
			print('use ResNet',self.cnn_type)

		elif self.model_type == ResCNN.DenseNet.value: # DenseNet
			if cnn_type == 121:
				self.sel_feat_layer = {1, 4, 6, 11} # conv1, block1, block2
				self.model = models.densenet121(pretrained=True)
			else:
				self.cnn_type = 161
				self.sel_feat_layer = {0, 4, 6, 11} # conv1, block1, block2
				self.model = models.densenet161(pretrained=True)
			print('use DenseNet',self.cnn_type)

		elif self.model_type == ResCNN.WideResNet.value: # wide_resnet
			if cnn_type == 101:
				self.model = models.wide_resnet50_2(pretrained=True)
			else:
				self.cnn_type = 50
				self.model = models.wide_resnet50_2(pretrained=True)
			print('use Wild_ResNet',self.cnn_type)

		elif self.model_type == ResCNN.ResNext.value: # resnext
			if cnn_type == 101:
				self.model = models.resnext101_32x8d(pretrained=True)
			else:
				self.cnn_type = 50
				self.model = models.resnext50_32x4d(pretrained=True)
			print('use ResNext',self.cnn_type)

		self.count = 0
		if use_gpu:
			print('use cuda model')
			self.model = self.model.cuda()
		self.model.eval()
		if self.model_type == ResCNN.DenseNet.value: # dense layer
			features = list(self.model.features)[:]
			self.features = nn.ModuleList(features).eval()

		# using Grad-CAM with softmax weight
		params = list(self.model.parameters())
		self.weight_softmax = np.squeeze(params[-2].data.numpy())
		self.CAMs = None

		# test feature
		if self.use_layer == 0:  # run combine cnn_features
			self.use_layer = -4
			print('test linear combined cnn_features')
		elif self.use_layer > 4:  # run combine features with grad-CAM
			print('test combined cnn_features with grad-CAM')
		else:  # test each conv layers
			print('test single conv layer', self.use_layer)
	
	def forward(self, x):
		feat = []
		if self.model_type != ResCNN.DenseNet.value: # ResNet, Wild_ResNet, ResNext
			x = self.model.conv1(x)
			feat.append(x.cpu().numpy())
			x = self.model.bn1(x)
			x = self.model.relu(x)
			x = self.model.maxpool(x)
			if self.use_layer == 1: return x,feat
			
			x = self.model.layer1(x)
			feat.append(x.cpu().numpy())
			if self.use_layer == 2: return x, feat
			
			x = self.model.layer2(x)
			feat.append(x.cpu().numpy())
			if self.use_layer == 3: return x, feat
			x = self.model.layer3(x)
			x = self.model.layer4(x)
			feat.append(x.cpu().numpy()) # feat CAM
			# GAP + FC
			x = self.model.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.model.fc(x)
		
		else: # DenseNet
			for ii, model in enumerate(self.features):
				x = model(x)
				# print(ii, "x: ", x.shape)
				if ii in self.sel_feat_layer:
					#print("get featture: ", ii, x.shape)
					feat.append(x.cpu().numpy())
					if self.use_layer == len(feat):
						return x,feat
			x = F.relu(x, inplace=True)
			x = F.adaptive_avg_pool2d(x, (1, 1))
			x = torch.flatten(x, 1)
			x = self.model.classifier(x)

		return x, feat

	def get_cnn_feat(self, img, feat_visual=False, use_CAM=False, save_CAM=False):
		height, width = 224,224
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_tensor = preprocess(img)
		img_tensor = img_tensor.unsqueeze(0)
		if use_gpu:
			img_tensor = img_tensor.cuda()
		
		# get features
		with torch.no_grad():
			out,feat = self.forward(img_tensor)
			if feat_visual and self.use_layer>2:
				print('feat shape:', feat[0].shape,feat[1].shape,feat[2].shape)
				if self.model_type == ResCNN.DenseNet.value:
					draw_features(8, 8, feat[0], save_feat + "DenseNet_conv1_112.png")
					draw_features(10,10,feat[1], save_feat + "DenseNet_block1_56.png")
					# draw_features(10,10,feat[2], save_feat + "DenseNet_block2_28.png")
				else:
					draw_features(8, 8, feat[0], save_feat + "ResNet_conv1_112.png")
					draw_features(10,10,feat[1], save_feat + "ResNet_layer1_56.png")
					draw_features(10,10,feat[2], save_feat + "ResNet_layer2_28.png")
				# self.count += 1
			
			if use_CAM:
				print('feat_cam: ', feat[-1].shape, 'logit out: ', out.shape)
				# get class probs
				h_x = F.softmax(out, dim=1).data.squeeze()
				print('h_x: ', h_x.shape)
				probs, idx = h_x.sort(0, True)
				if use_gpu:
					probs, idx = probs.cpu(), idx.cpu()
				probs = probs.numpy()
				idx = idx.numpy()
				# run Grad_CAM with top 1 class
				self.CAMs = returnCAM(feat[-1], self.weight_softmax, [idx[0]])
				heatmap = cv2.applyColorMap(cv2.resize(self.CAMs[0],(width, height)), cv2.COLORMAP_JET)
				if save_CAM:
					img.save(save_img + 'Res_test.jpg')
					img = cv2.imread(save_img + 'Res_test.jpg')
					result = heatmap * 0.3 + img * 0.5
					cv2.imwrite(save_img + 'Res_CAM.jpg', result)
					print('saved Res CAM')

			if self.use_layer <= 0: # linear combined cnn_features
				# use conv layer 1,3,4
				feat.pop(1)
				return feat
			elif self.use_layer > 4:
				# use conv layer 1,2,3 + grad-CAM
				feat.append(self.CAMs[0])
				return feat
			else: # test single feat layer
				print(feat[self.use_layer-1].shape)
				return feat[self.use_layer-1]
