from PIL import Image
import scipy.io as scio
import cv2
import numpy as np
import os
import pickle
import cupoch as cph

if __name__ == '__main__':
    mode = 'test'
    root = './datasets/ycb/YCB_Video_Dataset'

    if mode == 'train':
        path = 'datasets/ycb/dataset_config/train_data_temporal_6.txt'
        # self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
    elif mode == 'test':
        path = 'datasets/ycb/dataset_config/test_data_temporal_6.txt'
    list = []

    input_file = open(path, 'r')
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        list.append(input_line)
    input_file.close()

    source_path = None
    last_data_folder = ''

    for index in range(len(list)):
        data_folder = list[index].split('/')[1]
        if source_path is not None:
            meta = scio.loadmat('{0}/{1}-meta.mat'.format(root, list[index]))

            focal_length = meta.get("intrinsic_matrix")[0][0]
            principal_point = [meta.get("intrinsic_matrix")[0][2], meta.get("intrinsic_matrix")[1][2]]

            pinhole_camera_intrinsic = cph.io.read_pinhole_camera_intrinsic(
                "{0}/{1}".format(root, "camera_primesense.json"))
            pinhole_camera_intrinsic.set_intrinsics(640, 480, meta.get("intrinsic_matrix")[0][0],
                                                    meta.get("intrinsic_matrix")[1][1],
                                                    meta.get("intrinsic_matrix")[0][2],
                                                    meta.get("intrinsic_matrix")[1][2])

            target_path = list[index]
            source_color = cph.io.read_image("{0}/{1}-color.png".format(root, source_path))
            target_color = cph.io.read_image("{0}/{1}-color.png".format(root, target_path))
            source_depth = cph.io.read_image("{0}/{1}-depth.png".format(root, source_path))
            target_depth = cph.io.read_image("{0}/{1}-depth.png".format(root, target_path))
            source_rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(
                source_color, source_depth, depth_scale=int(meta.get("factor_depth")))
            target_rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(
                target_color, target_depth, depth_scale=int(meta.get("factor_depth")))
            target_pcd = cph.geometry.PointCloud.create_from_rgbd_image(
                target_rgbd_image, pinhole_camera_intrinsic)

            option = cph.odometry.OdometryOption()
            odo_init = np.identity(4)
            # print(option)

            if data_folder == last_data_folder:
                [success_hybrid_term, trans_hybrid_term,
                 info] = cph.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
                    odo_init, cph.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

                output_file = open('{0}/{1}-odom-hybrid.pickle'.format(root, list[index]), 'wb')
                pickle.dump(trans_hybrid_term, output_file)
                print('outputting {0}'.format('{0}/{1}-odom.pickle'.format(root, list[index])))
                output_file.close()
            else:
                print('changing dir from {0} to {1}'.format(last_data_folder, data_folder))

            last_data_folder = data_folder
            source_path = target_path
        else:
            last_data_folder = data_folder
            source_path = list[index]
