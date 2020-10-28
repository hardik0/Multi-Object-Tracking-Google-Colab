# orignal code: https://github.com/CaptainEven/FairMOTVehicle/blob/master/src/gen_labels_detrac.py

import os
import shutil
import copy
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm


def preprocess(src_root, dst_root):
    """
    :param src_root:
    :param dst_root:
    :return:
    """
    if not os.path.isdir(src_root):
        print("[Err]: invalid source root")
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print("{} made".format(dst_root))

    # Create a directory structure for training MOT
    dst_img_dir_train = dst_root + '/images/train'
    dst_img_dir_test = dst_root + '/images/test'
    dst_labels_with_ids = dst_root + '/labels_with_ids'
    if not os.path.isdir(dst_img_dir_train):
        os.makedirs(dst_img_dir_train)
    if not os.path.isdir(dst_img_dir_test):
        os.makedirs(dst_img_dir_test)
    if not os.path.isdir(dst_labels_with_ids):
        os.makedirs(dst_labels_with_ids)

    # Traverse src_root, further improve the training directory and copy files
    for x in os.listdir(src_root):
        x_path = src_root + '/' + x
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                if y.endswith('.jpg'):
                    y_path = x_path + '/' + y
                    if os.path.isfile(y_path):
                        # Create image target directory for training
                        dst_img1_dir = dst_img_dir_train + '/' + x + '/img1'
                        if not os.path.isdir(dst_img1_dir):
                            os.makedirs(dst_img1_dir)

                        # copy image to train image dir
                        shutil.copy(y_path, dst_img1_dir)
                        print('{} cp to {}'.format(y, dst_img1_dir))


def draw_ignore_regions(img, boxes):
    """
    Input picture ignore regions blacked out
    :param img: opencv(numpy array): H×W×C
    :param boxes: a list of boxes: left(box[0]), top(box[1]), width(box[2]), height(box[3])
    :return:
    """
    if img is None:
        print('[Err]: Input image is none!')
        return -1

    for box in boxes:
        box = list(map(lambda x: int(x + 0.5), box))  # 四舍五入
        img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = [0, 0, 0]

    return img


def gen_labels(xml_root, img_root, label_root, viz_root=None):
    """
    Parse xml (parse the result and visualize) + generate labels
    :param xml_root:
    :param img_root:
    :param label_root:
    :param viz_root:
    :return:
    """
    if not (os.path.isdir(xml_root) and os.path.isdir(img_root)):
        print('[Err]: invalid dirs')
        return -1

    xml_f_paths = [xml_root + '/' + x for x in os.listdir(xml_root)]
    img_dirs = [img_root + '/' + x for x in os.listdir(img_root)]
    xml_f_paths.sort()  # File names are sorted naturally
    img_dirs.sort()  # Natural sorting of directory names

    assert (len(xml_f_paths) == len(img_dirs))

    # Record the id at the beginning of each sequence, initialized to 0
    track_start_id = 0

    # Record the total number of frames
    frame_cnt = 0

    # Traverse each video seq (each seq corresponds to an xml file and an img_dir)
    for x, y in zip(xml_f_paths, img_dirs):
        if os.path.isfile(x) and os.path.isdir(y):
            if x.endswith('.xml'):  # 找到了xml原始标注文件
                sub_dir_name = os.path.split(y)[-1]
                if os.path.split(x)[-1][:-4] != sub_dir_name:
                    print('[Err]: xml file and dir not match')
                    continue

                # ----- Process this video seq
                # Read and parse xml
                tree = ET.parse(x)
                root = tree.getroot()
                seq_name = root.get('name')  # video sequence name (subdirectory name)
                if seq_name != sub_dir_name:
                    print('[Warning]: xml file and dir not match')
                    continue
                print('Start processing seq {}...'.format(sub_dir_name))

                # Create a training label subdirectory of the video seq
                seq_label_root = label_root + '/' + seq_name + '/img1/'
                if not os.path.isdir(seq_label_root):
                    os.makedirs(seq_label_root)
                else:  # If it already exists, first clear the original data and recreate the recursive directory
                    shutil.rmtree(seq_label_root)
                    os.makedirs(seq_label_root)

                # Record the maximum track_id of the video seq
                seq_max_tar_id = 0

                # Create seq_label_root
                seq_label_root = label_root + '/' + seq_name + '/img1'
                if not os.path.isdir(seq_label_root):
                    os.makedirs(seq_label_root)

                # Find ignored_region (used to black out the rectangular area corresponding to the original image)
                ignor_region = root.find('ignored_region')

                # Record all boxes in the ignored_region of the video seq
                boxes = []
                for box_info in ignor_region.findall('box'):
                    box = [float(box_info.get('left')),
                           float(box_info.get('top')),
                           float(box_info.get('width')),
                           float(box_info.get('height'))]
                    # print('left {:.2f}, top {:.2f}, width {:.2f}, height {:.2f}'
                    #       .format(box[0], box[1], box[2], box[3]))
                    boxes.append(box)

                # Traverse each frame
                for frame in root.findall('frame'):
                    # Update frame statistics
                    frame_cnt += 1

                    target_list = frame.find('target_list')
                    targets = target_list.findall('target')
                    density = int(frame.get('density'))
                    if density != len(targets):  # 处理这一帧的每一个目标
                        print('[Err]: density not match @', frame)
                        return -1
                    # print('density {:d}'.format(density))

                    # ----- Process the current frame
                    # Get the frame id of the current frame in this video seq
                    f_id = int(frame.get('num'))

                    # ----- Read a frame of the video seq, and blacken the corresponding region (ignore_region under the box)
                    img_path = y + '/img1/img{:05d}.jpg'.format(f_id)
                    if not os.path.isfile(img_path):
                        print('[Err]: image file not exists!')
                        return -1
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # H×W×C
                    if img is None:  # channels: BGR
                        print('[Err]: read image failed!')
                        return -1

                    # Blacken the picture and write it to the original training directory path
                    img = draw_ignore_regions(img, boxes)
                    cv2.imwrite(img_path, img)

                    # If the visualization directory is not empty, perform visualization calculations
                    if not (viz_root is None):
                        # Picture visualization directory and path
                        viz_path = viz_root + '/' + seq_name + '_' + os.path.split(img_path)[-1]

                        # Deep copy an img data as a visual output
                        img_viz = copy.deepcopy(img)

                    # Record each line label_str of the frame (corresponding to a detection or tracking target)
                    frame_label_strs = []

                    # Traverse each target (object) in this frame
                    for target in targets:
                        # Read each target and write label_with_id in append
                        target_id = int(target.get('id'))

                        # Record the largest target id of the video seq
                        if target_id > seq_max_tar_id:
                            seq_max_tar_id = target_id

                        # Record the track id of the target (object) (starting from 1)
                        track_id = target_id + track_start_id

                        # Read the bbox corresponding to the target
                        bbox_info = target.find('box')
                        bbox_left = float(bbox_info.get('left'))
                        bbox_top = float(bbox_info.get('top'))
                        bbox_width = float(bbox_info.get('width'))
                        bbox_height = float(bbox_info.get('height'))

                        # Read the attributes corresponding to the target (only the attributes of interest are listed here for the time being)
                        attr_info = target.find('attribute')
                        vehicle_type = str(attr_info.get('vehicle_type'))
                        trunc_ratio = float(attr_info.get('truncation_ratio'))

                        # Calculate the results of the visualization here
                        if not (viz_root is None):  # If the visualization directory is not empty
                            # Draw bbox for target
                            pt_1 = (int(bbox_left + 0.5), int(bbox_top + 0.5))
                            pt_2 = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))
                            cv2.rectangle(img_viz,
                                          pt_1,
                                          pt_2,
                                          (0, 255, 0),
                                          2)
                            # Draw attribute text
                            veh_type_str = 'Vehicle type: ' + vehicle_type
                            veh_type_str_size = cv2.getTextSize(veh_type_str,
                                                                cv2.FONT_HERSHEY_PLAIN,
                                                                1.3,
                                                                1)[0]
                            cv2.putText(img_viz,
                                        veh_type_str,
                                        (pt_1[0],
                                         pt_1[1] + veh_type_str_size[1] + 8),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.3,
                                        [225, 255, 255],
                                        1)
                            tr_id_str = 'Vehicle ID: ' + str(track_id)
                            tr_id_str_size = cv2.getTextSize(tr_id_str,
                                                             cv2.FONT_HERSHEY_PLAIN,
                                                             1.3,
                                                             1)[0]
                            cv2.putText(img_viz,
                                        tr_id_str,
                                        (pt_1[0],
                                         pt_1[1] + veh_type_str_size[1] + tr_id_str_size[1] + 8),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.3,
                                        [225, 255, 255],
                                        1)
                            trunc_str = 'Trunc ratio {:.2f}'.format(trunc_ratio)
                            trunc_str_size = cv2.getTextSize(trunc_str, cv2.FONT_HERSHEY_PLAIN, 1.3, 1)[0]
                            cv2.putText(img_viz,
                                        trunc_str,
                                        (pt_1[0],
                                         pt_1[1] + veh_type_str_size[1]
                                         + tr_id_str_size[1] + trunc_str_size[1] + 8),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.3,
                                        [225, 255, 255],
                                        1)

                        # ----- Output label
                        # Calculate the center point coordinates of bbox
                        bbox_center_x = bbox_left + bbox_width * 0.5
                        bbox_center_y = bbox_top + bbox_height * 0.5

                        # Normalize bbox ([0., 1.])
                        bbox_center_x /= img.shape[1]
                        bbox_center_y /= img.shape[0]
                        bbox_width /= img.shape[1]
                        bbox_height /= img.shape[0]

                        # Organize the content of the label, TODO: optimize IO, read and write once on the hard disk, and output the label after each frame
                        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            track_id,
                            bbox_center_x,  # center_x
                            bbox_center_y,  # center_y
                            bbox_width,  # bbox_w
                            bbox_height)  # bbox_h
                        frame_label_strs.append(label_str)

                        # # Output label
                        # label_f_path = seq_label_root + '/img{:05d}.txt'.format(f_id)
                        # with open(label_f_path, 'a') as f:
                        #     f.write(label_str)

                    # Output visualization results
                    if not (viz_root is None):  # If the visualization directory is not empty
                        cv2.imwrite(viz_path, img_viz)

                    # ----- The targets of this frame are output only once after parsing
                    # Output label
                    label_f_path = seq_label_root + '/img{:05d}.txt'.format(f_id)
                    with open(label_f_path, 'w') as f:
                        for label_str in frame_label_strs:
                            f.write(label_str)

                    # print('frame {} in seq {} processed done'.format(f_id, seq_name))

                # After processing the video seq, update track_start_id
                print('Seq {} start track id: {:d}, has {:d} tracks'
                      .format(seq_name, track_start_id + 1, seq_max_tar_id))
                track_start_id += seq_max_tar_id
                print('Processing seq {} done.\n'.format(sub_dir_name))

    print('Total {:d} frames'.format(frame_cnt))


def gen_dot_train_file(data_root, rel_path, out_root):
    """
    Generate .train file
    :param data_root:
    :param rel_path:
    :param out_root:
    :return:
    """
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('[Err]: invalid root')
        return

    out_f_path = out_root + '/detrac.train'
    cnt = 0
    with open(out_f_path, 'w') as f:
        root = data_root + rel_path
        seqs = [x for x in os.listdir(root)]
        seqs.sort()
        for seq in tqdm(seqs):
            img_dir = root + '/' + seq + '/img1'
            imgs = [x for x in os.listdir(img_dir)]
            imgs.sort()
            for img in imgs:
                if img.endswith('.jpg'):
                    img_path = img_dir + '/' + img
                    if os.path.isfile(img_path):
                        item = img_path.replace(data_root + '/', '')
                        print(item)
                        f.write(item + '\n')
                        cnt += 1

    print('Total {:d} images for training'.format(cnt))


def find_file_with_suffix(root, suffix, f_list):
    """
    Recursively find specific suffix files
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            find_file_with_suffix(f_path, suffix, f_list)


def count_files(img_root, label_root):
    """
    Count the total number of pictures and label txt files
    :param img_root:
    :param label_root:
    :return:
    """
    img_file_list, label_f_list = [], []

    find_file_with_suffix(img_root, '.jpg', img_file_list)
    find_file_with_suffix(label_root, '.txt', label_f_list)

    print('Total {:d} image files'.format(len(img_file_list)))
    print('Total {:d} label(txt) files'.format(len(label_f_list)))


def clean_train_set(img_root, label_root):
    """
    Clean up the problem that the number of pictures does not match the number of label files
    :param img_root:
    :param label_root:
    :return:
    """
    if not (os.path.isdir(img_root) and os.path.isdir(label_root)):
        print('[Err]: incalid root!')
        return

    img_dirs = [img_root + '/' + x for x in os.listdir(img_root)]
    label_dirs = [label_root + '/' + x for x in os.listdir(label_root)]

    assert (len(img_dirs) == len(label_dirs))

    # Sort by video seq name
    img_dirs.sort()
    label_dirs.sort()

    for img_dir, label_dir in tqdm(zip(img_dirs, label_dirs)):
        # Check of a couple
        for img_name in os.listdir(img_dir + '/img1'):
            # print(img_name)
            txt_name = img_name.replace('.jpg', '.txt')
            txt_path = label_dir + '/img1/' + txt_name
            img_path = img_dir + '/img1/' + img_name
            if os.path.isfile(img_path) and os.path.isfile(txt_path):
                continue  # 两者同时存在, 无需处理
            elif os.path.isfile(img_path) and (not os.path.isfile(txt_path)):
                os.remove(img_path)
                print('{} removed.'.format(img_path))
            elif os.path.isfile(txt_path) and (not os.path.isfile(img_path)):
                os.remove(txt_path)
                print('{} removed.'.format(txt_path))

if __name__ == '__main__':
    preprocess(src_root='/content/Towards-Realtime-MOT/data/Insight-MVT_Annotation_Train',
               dst_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC')

    gen_labels(xml_root='/content/Towards-Realtime-MOT/data/DETRAC-Train-Annotations-XML',
               img_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC/images/train',
               label_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC/labels_with_ids/train',
               viz_root='/content/Towards-Realtime-MOT/data/viz_result')

    clean_train_set(img_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC/images/train',
                    label_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC/labels_with_ids/train')

    gen_dot_train_file(data_root='/content/Towards-Realtime-MOT/data/MOT',
                       rel_path='/DETRAC/images/train',
                       out_root='/content/Towards-Realtime-MOT/data')
    
    count_files(img_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC/images/train',
                label_root='/content/Towards-Realtime-MOT/data/MOT/DETRAC/labels_with_ids')

    print('Done')
