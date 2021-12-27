import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import torch
from lib.knn import KNearestNeighbor
from torch_batch_svd import svd


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[2]

    # translate points to their centroids
    centroid_A = torch.mean(A, dim=1).unsqueeze(1).repeat(1, A.shape[1], 1)
    centroid_B = torch.mean(B, dim=1).unsqueeze(1).repeat(1, B.shape[1], 1)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = torch.bmm(AA.transpose(1, 2), BB)
    U = torch.empty(A.shape[0], m, m, device="cuda")
    S = torch.empty(A.shape[0], m, device="cuda")
    Vt = torch.empty(A.shape[0], m, m, device="cuda")

    U, S, Vt = svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    # translation
    t = centroid_B[:, 0, :] - torch.bmm(R, centroid_A[:, 0, :].unsqueeze(2)).squeeze(2)

    # homogeneous transformation
    T = torch.eye(m + 1, device="cuda").repeat(A.shape[0], 1, 1)
    T[:, :m, :m] = R
    T[:, :m, m] = t

    return T, R, t


def my_nearest_neighbor_weak(dst, src):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    b = dst.shape[0]
    n = dst.shape[1]
    m = src.shape[1]
    d = dst.shape[2]

    dist = torch.pow(dst.unsqueeze(2).expand(b, n, m, d) - src.unsqueeze(1).expand(b, n, m, d), 2).sum(3)

    inds_ext_f = torch.argmin(dist, dim=1)

    dst_ext_f = dst.gather(1, inds_ext_f.unsqueeze(2).repeat(1, 1, d))
    dis_ext_f = torch.norm(dst_ext_f - src, dim=2)
    
    return dis_ext_f, inds_ext_f, dst_ext_f


def my_icp_weak(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    global T, mean_error
    b = init_pose.shape[0]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = torch.ones((m + 1, A.shape[0])).cuda()
    dst = torch.ones((m + 1, B.shape[0])).cuda()
    src[:m, :] = A.t()
    dst[:m, :] = B.t()
    src_ori = None
    # apply the initial pose estimation
    if init_pose is not None:
        src = torch.bmm(init_pose, src.unsqueeze(0).repeat(init_pose.shape[0], 1, 1))
        dst = dst.unsqueeze(0).repeat(init_pose.shape[0], 1, 1)
    else:
        print("init pose is None!!!")
    result_num = src.shape[0]
    total_T = torch.eye(m + 1, device="cuda").unsqueeze(0).repeat(b, 1, 1)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points

        distances, indices, dst_ret = my_nearest_neighbor_weak(dst[:, :m, :].transpose(1, 2),
                                                               src[:, :m, :].transpose(1, 2))

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:, :m].transpose(1, 2), dst_ret)

        # update the current source
        src = torch.bmm(T, src)
        total_T = torch.bmm(T, total_T)
        # check error
        mean_error = torch.mean(distances, dim=1)

        # calculate final transformation
    return total_T, mean_error, src[:, :m, :]