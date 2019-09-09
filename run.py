# old version = 8.12
import cv2
from tracker import KCFTracker

env = 'MAC'
test = 'BenchMark' # 'Cam'

if env == 'MAC':
	test_seq = '/Users/apple/Downloads/cnnKCF_proj_code/KCF_8.12/Sequence/Boy/'
elif env == 'WIN':
	test_seq = 'Sequence\\Boy\\'

# get IoU of track_bbox and truth_bbox
def IoU(box1, box2): # 2D bounding box [top, left, bottom, right]
	in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
	in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
	inter = 0 if in_h<0 or in_w<0 else in_h*in_w
	union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])
	union -= inter
	iou = inter / union
	return iou

# read groundtruth_rect.txt from seq_file to get bbox
def read_groundtruth(seq_path):
	truth_bbox = []
	with open(seq_path+'groundtruth_rect.txt', 'r') as file_to_read:
		while True:
			lines = file_to_read.readline() # 整行读取数据
			if not lines:
				break
			p_tmp = [int(i) for i in lines.split(',')]
			p_tmp = tuple(p_tmp)
			truth_bbox.append(p_tmp)
	return truth_bbox


def run_tracker(frame, truth_bbox, seq_val=True):
	# KCF tracker use (hog, fixed_Window, multi_scale, cnn)
	tracker = KCFTracker(False, True, False, True)
	count = 1
	if seq_val == False:
		cam = cv2.VideoCapture(0)
		tracker.init(truth_bbox, frame)
	elif seq_val == True:
		tracker.init(truth_bbox[0], frame)
		frame_num = len(truth_bbox)

	while True:
		if seq_val == False:
			ok, frame = cam.read()
			if ok == False: break
		elif seq_val == True:
			count += 1
			if count > frame_num: break
			# read img from seq_file
			if count < 10:
				img_path = test_seq + 'img/000'+str(count)+'.jpg'
			elif count < 100:
				img_path = test_seq + 'img/00' +str(count)+'.jpg'
			else:
				img_path = test_seq + 'img/0'  +str(count)+'.jpg'
			frame = cv2.imread(img_path)

		timer = cv2.getTickCount()
		bbox = tracker.update(frame)
		bbox = list(map(int, bbox))
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

		# Tracking success
		p1 = (int(bbox[0]), int(bbox[1])) # top,left
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) # top+x,left+y = bottom,right
		# draw tracking result
		cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
		if seq_val == True:
			t1 = (int(truth_bbox[count-1][0]), int(truth_bbox[count-1][1]))
			t2 = (int(truth_bbox[count-1][0] + truth_bbox[count-1][2]), int(truth_bbox[count-1][1] + truth_bbox[count-1][3]))
			# draw ground_truth bbox
			cv2.rectangle(frame, t1, t2, (0, 255, 0), 2, 1)
			# get center of ground_truth bbox, get displacement!!
			tcx = truth_bbox[count-1][0] + truth_bbox[count-1][2] /2.0
			tcy = truth_bbox[count-1][1] + truth_bbox[count-1][3] /2.0
			tcx_pre = truth_bbox[count-2][0] + truth_bbox[count-2][2] /2.0
			tcy_pre = truth_bbox[count-2][1] + truth_bbox[count-2][3] /2.0
			print('ground_truth:',tcx,tcy,'prev:',tcx_pre,tcy_pre,' ; displacement:', tcx-tcx_pre,tcy-tcy_pre)
			
			# using re_init when tracking failed, IoU<=0.5
			#box1,box2 = [p1[0],p1[1],p2[0],p2[1]], [t1[0],t1[1],t2[0],t2[1]]
			#if IoU(box1,box2) <= 0.5:
			#	tracker.init(truth_bbox[count-1], frame)
			#	print('###########\nTrack Fail in frame',count,'\nRe_init KCF\n###########!')
		
		# Put FPS
		cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

		cv2.imshow("Tracking", frame)

		# Exit if ESC pressed
		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break
	
	if seq_val == False:
		cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	if test == 'CAM':
		video = cv2.VideoCapture(0)
		ok, frame = video.read()
		print(ok, frame.shape)
		# cv2.selectROI to get init bbox on frame(0)
		bbox = cv2.selectROI('Select ROI', frame, False)
		if min(bbox) == 0: exit(0)
		# run with CAM
		run_tracker(frame, bbox, False)

	elif test == 'BenchMark':
		# test CNN model in tracker.py
		# AlexNet,GoogLeNet,VggNet, ResNet,DenseNet,WideResNet,ResNext,
		#SqueezeNet,MobileNet,ShuffleNet,MnasNet, SENet
		truth_bbox = read_groundtruth(test_seq)
		frame = cv2.imread(test_seq+'img/0001.jpg')
		run_tracker(frame, truth_bbox, True)
