"""
X-ray patcher module.
Need kornia
"""

import numpy as np
import torch
import kornia
import os
import cv2
import random

PARAM_IRON = [
    [0, 0, 104.3061111111111], 
    [-199.26894460884833, -1.3169138497713286, 227.17803542009827], 
    [-21.894450101465132, 0.20336113292167177, 274.63740523563814]
]

PARAM_IRON_FIX = [
    [0, 0, 104.3061111111111], 
    [0, 0, 226.1507], 
    [0, 0, 225.2509],
]


PARAM_PLASTIC = [
    [0, 0, 16.054857142857145], 
    [-175.96004580018538, -0.02797999280157535, 226.59010365257998], 
    [-1.1977592197679745, 0.03212775118421846, 251.99895369583868]
]

PARAM_GLASS = [
    [0, 0, 44.9139739229025], 
    [-162.0511635446029, -0.1537546525499077, 169.87370033743895],
    [68.9094475565913, -0.14688815701438654, 174.05450704994433]
]

MIN_EPS = 1e-6

def load_infos(xlist):
    vertices = []
    faces = []

    for elm in xlist:
        if elm[0] == "v" and elm[1] != 'n':
            vertices.append([float(elm.split(" ")[1]), float(elm.split(" ")[2]), float(elm.split(" ")[3])])
        elif elm[0] == "f":
            if '//' in elm:
                faces.append([int(elm.split(" ")[1][0]) - 1, int(elm.split(" ")[2][0]) - 1, int(elm.split(" ")[3][0]) - 1])
            else:
                faces.append([int(elm.split(" ")[1]) - 1, int(elm.split(" ")[2]) - 1, int(elm.split(" ")[3]) - 1])

    vertices = torch.Tensor(vertices).type(torch.cuda.FloatTensor)
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces

def load_from_file(obj, root, M=None):
    """
    Load vertices and faces from an .obj file.
    coordinates will be normalized to N(0.5, 0.5)
    """
    ## single part
    if type(obj) == dict:
        name = list(obj.keys())[0]
        path = root + name + '.obj'
        with open(path, "r") as fp:
            xlist = fp.readlines()

        vertices, faces = load_infos(xlist)
        # rotate
        if M is not None:
            vertices = torch.mm(vertices, M)
    
        # clamp
        min_xyz, _ = torch.min(vertices, 0)
        min_xyzall = min_xyz.repeat((vertices.shape[0], 1))
        max_xyz, _ = torch.max(vertices, 0)
        max_xyzall = max_xyz.repeat((vertices.shape[0], 1))
        max_len = max_xyz - min_xyz
        max_len[-1] = 0
        max_len2, _ = torch.max(max_len, 0)
        # vertices = (vertices - min_xyzall) / max_len2
        len = max_xyzall - min_xyzall
        len[:, 0] = max_len2
        len[:, 1] = max_len2
        # vertices = (vertices - min_xyzall) / (max_xyzall - min_xyzall)
        vertices = (vertices - min_xyzall) / len
        
        faces = np.array(faces, dtype=np.int32)
        return [vertices], [faces]
    ## multi parts
    else:
        whole_name = obj[0]
        whole_path = root + whole_name + '.obj'
        with open(whole_path, "r") as fp:
            xlist = fp.readlines()
        whole_vertices, whole_faces = load_infos(xlist)

        vertices = []
        faces = []
        names = obj[-1].keys()
        for name in names:
            path = root + name + '.obj'
            with open(path, "r") as fp:
                xlist = fp.readlines()
            vertice, face = load_infos(xlist)
            vertices.append(vertice)
            faces.append(face)

        # rotate
        if M is not None:
            whole_vertices = torch.mm(whole_vertices, M)
            vertices = [torch.mm(v, M) for v in vertices]
        
        # for each v do same clamp
        min_xyz, _ = torch.min(whole_vertices, 0)
        # min_xyzall = min_xyz.repeat((whole_vertices.shape[0], 1))
        max_xyz, _ = torch.max(whole_vertices, 0)
        # max_xyzall = max_xyz.repeat((whole_vertices.shape[0], 1))
        max_len = max_xyz - min_xyz
        len = max_len.clone()
        max_len[-1] = 0
        max_len2, _ = torch.max(max_len, 0)
        len[0:2] = max_len2
        # vertices = (vertices - min_xyzall) / max_len2
        vertices_clamp = []
        for v in vertices:
            min_xyzall = min_xyz.repeat((v.shape[0], 1))
            len_all = len.repeat((v.shape[0], 1))
            
            vs = (v - min_xyzall) / len_all
            vertices_clamp.append(vs)
        
        return vertices_clamp, faces
        

def save_to_file(path, vertices, faces):
    """
    Save vertices and faces to an .obj file.
    """
    with open(path, "w") as fp:
        for i in range(vertices.shape[0]):
            fp.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(faces.shape[0]):
            fp.write("f {} {} {}\n".format(faces[i][0]+1, faces[i][1]+1, faces[i][2]+1))

def get_func_hsv(params):
    def func(x, a, b, c):
        return a * torch.exp(b * x) + c

    return lambda x: torch.cat((func(x, *params[0]), func(x, *params[1]), func(x, *params[2])), 1)

def simulate(img, material="iron"):
    """
    img: Tensor (N, 1, H, W) range(0, 1) depth image
    return: (N, 3, H, W) range(0, 1) rgb image
    """
    if material == "iron":
        max_depth = 8
        params = PARAM_IRON
    if material == "iron_fix":
        max_depth = 8
        params = PARAM_IRON_FIX
    elif material == "plastic":
        max_depth = 40
        params = PARAM_PLASTIC
    elif material == "glass":
        max_depth = 10
        params = PARAM_GLASS
    sim = get_func_hsv(params)
    img_xray = sim(img * max_depth)
    img_xray = torch.clamp(img_xray, 0, 255) / 255
    img_xray[0][0] = img_xray[0][0] * 255 * np.pi / 90
    img_xray = kornia.color.hsv_to_rgb(img_xray)
    img_xray = torch.flip(img_xray, [1])
    img_xray = torch.clamp(img_xray, 0, 1)
    img = torch.cat((img, img, img), 1)
    mask = (img!=0)
    return img_xray, mask

def get_rotate_matrix(param):
    return rotate_matrix([torch.Tensor([param[0]]), torch.Tensor([param[1]]), torch.Tensor([param[2]])])

def rotate_matrix(matrix):
    """
    Rotate vertices in a obj file.
    matrix: a three-element list [Rx, Ry, Rz], R for rotate degrees (angle system)
    """
    x = matrix[0] * np.pi / 180
    y = matrix[1] * np.pi / 180
    z = matrix[2] * np.pi / 180
    rx = torch.Tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ]).cuda()
    ry = torch.Tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ]).cuda()
    rz = torch.Tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ]).cuda()
    M = torch.mm(torch.mm(rx, ry), rz)
    return M

def is_in_triangle(point, tri_points):
    """
    Judge whether the point is in the triangle
    """
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < 1e-4:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1) & (inverDeno != 0)

def get_point_weight(point, tri_points):
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < 1e-4:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def are_in_triangles(points, tri_points):
    """
    Judge whether the points are in the triangles
    assume there are n points, m triangles
    points shape: (n, 2)
    tri_points shape: (m, 3, 2)
    """
    tp = tri_points
    n = points.shape[0]
    m = tp.shape[0]

    # vectors
    # shape: (m, 2)
    v0 = tp[:, 2, :] - tp[:, 0, :]
    v1 = tp[:, 1, :] - tp[:, 0, :]
    # shape: (n, m, 2)
    v2 = points.unsqueeze(1).repeat(1, m, 1) - tp[:, 0, :]

    # dot products
    # shape: (m, 2) =sum=> (m, 1)
    dot00 = torch.mul(v0, v0).sum(dim=1)
    dot01 = torch.mul(v0, v1).sum(dim=1)
    dot11 = torch.mul(v1, v1).sum(dim=1)
    # shape: (n, m, 2) =sum=> (n, m, 1)
    dot02 = torch.mul(v2, v0).sum(dim=2)
    dot12 = torch.mul(v2, v1).sum(dim=2)

    # barycentric coordinates
    # shape: (m, 1)
    inverDeno = dot00*dot11 - dot01*dot01
    zero = torch.zeros_like(inverDeno)
    inverDeno = torch.where(inverDeno < MIN_EPS, zero, 1 / inverDeno)

    # shape: (n, m, 1)
    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno
    
    w0 = 1 - u - v
    w1 = v
    w2 = u

    # check if point in triangle
    return (u >= -MIN_EPS) & (v >= -MIN_EPS) & (u + v <= 1+MIN_EPS) & (inverDeno != 0), w0, w1, w2

def ball2depth(vertices, faces, h, w):
    """
    Save obj file as a depth image, z for depth and x,y for position
    a ball with coord in [0, 1]
    h, w: the output image height and width
    return: a depth image in shape [h, w]
    """
    vertices = torch.clamp(vertices, 0, 1)
    vs = vertices.clone()
    vs[:, 0] = vertices[:, 0] * w
    vs[:, 1] = vertices[:, 1] * h
    vertices = vs
    faces = torch.LongTensor(faces).cuda()
    
    points = torch.Tensor([(i, j) for i in range(h) for j in range(w)]).cuda()
    tri_points = vertices[faces, :2]
    in_triangle, w0, w1, w2 = are_in_triangles(points, tri_points)
    
    point_depth = w0 * vertices[faces[:, 0], 2] + w1 * vertices[faces[:, 1], 2] + w2 * vertices[faces[:, 2], 2]
    
    min_depth = torch.min(torch.where(in_triangle, point_depth, torch.full_like(point_depth, 9999)), dim=1).values
    max_depth = torch.max(torch.where(in_triangle, point_depth, torch.full_like(point_depth, -9999)), dim=1).values

    # image = torch.clamp(max_depth - min_depth, 0, 1).view(h, w)
    image = max_depth - min_depth
    image = image / image.max()
    image = torch.clamp(image, 0, 1).reshape(h, w)
    
    return image

def cal_patch_poly(patch):
    """
    calculate coordinates of four vertices of rotate bounding box
    """
    patch = patch.cpu().detach().numpy().astype(np.uint8)
    patch = patch.transpose(1, 2, 0)
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 图像二值化
    binary = np.expand_dims(binary, axis=2)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找物体轮廓
    assert len(contours) == 1
    rect = cv2.minAreaRect(contours[0])
    points = cv2.boxPoints(rect)
    points = np.int0(points)

    return points

def find_stick_point(patch_h, patch_w, img_h, img_w, border=10):
    """
    Find stick point randomly
    """
    x_min = border
    y_min = border
    x_max = img_h - patch_h - border
    y_max = img_w - patch_w - border
    assert x_max > x_min and y_max > y_min
    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)
    return (x, y)

def save_img(path, img_tensor, shape):
    img_tensor = img_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.transpose(1, 2, 0)
    img = cv2.resize(img, (shape[1], shape[0]))
    cv2.imwrite(path, img)


if __name__ == '__main__':
    ## config
    obj_root = './objs/'                                    # obj root path
    depth_save_path = './depthes'                           # depth save path
    patch_save_path = './patches'                           # patch save path
    obj_dict = {                                            # .obj file name and infos
        'scissors_vf': {
                'category':'knife', 
                'multi-part': False,
                'material':'iron',
                'patch_size': (150, 150)
            }
    }                                                       

    if not os.path.exists(patch_save_path):
        os.mkdir(patch_save_path)

    
    rotate_param = [110, 38, 56]                             # rotate config (x, y, z)
    M = get_rotate_matrix(rotate_param)

    vertices, faces = load_from_file(obj_dict, obj_root, M) # load obj vertices and faces with rotation

    obj_dict_info = obj_dict

    print('------------Part (2): Generate pacthed image.------------')

    print('------------Step (1): Generate depth image.------------')
    # we only consider object has one part 
    v = vertices[0]
    f = faces[0]
    name = list(obj_dict_info.keys())[0]
    obj_infos = list(obj_dict_info.values())[0]

    patch_size = obj_infos['patch_size']
    group = torch.clamp(v, 0, 1)
    depth_clamp = ball2depth(group, f, patch_size[0], patch_size[1]).unsqueeze(0)

    # # Enlarge the pixel to the normal image range
    # depth_img = depth_clamp[:,:] * 255
    # # print(depth_img.shape)
    # save_img(depth_save_path+'/' + name + '_depth.png',depth_img, patch_size) 

    print('------------------Step (2): Generate patch.------------------')
    material = obj_infos['material']
    patch, mask = simulate(depth_clamp.unsqueeze(0), material)
    patch[~mask] = 1

    # # Enlarge the pixel to the normal image range
    # patch_img = patch.squeeze(0) * 255
    # save_img(patch_save_path+'/' + name + '_patch.png',patch_img, patch_size)

    
    ## Step (4): stick patch to image
    print('------------------Step (3): Stick patch.------------------')     
    img_root = '/media/datasets/rotate_labels_buaa/train/images/'
    patched_img_root = '/media/datasets/rotate_labels_buaa/train/images_patched/'
    
    img_id = 'train00001'
    
    if not os.path.exists(patched_img_root):
        os.mkdir(patched_img_root)
    
    ## load img
    print(img_id)
    img_path = os.path.join(img_root, img_id+'.jpg')
    img = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    img_tensor = img_tensor.cuda()
    
    img_h, img_w = img_tensor.shape[1:]
    point = find_stick_point(patch_size[0], patch_size[1], img_h, img_w)
    # stick patch
    img_tensor[:, point[0]:point[0]+patch_size[0], point[1]:point[1]+patch_size[1]].mul_(patch.squeeze(0))
    # save sticked img
    # save_img_name = img_id.split('.')[0] + '_' + name + '.jpg'
    new_img_path = os.path.join(patched_img_root, img_id+'.jpg')
    save_img(new_img_path, img_tensor, img_tensor.shape[1:])

    print('------------------part (3): Save new annotation.------------------') 
    ann_root = '/media/datasets/rotate_labels_buaa/train/annotations'
    patched_ann_root = '/media/datasets/rotate_labels_buaa/train/annotations_patched/'
    if not os.path.exists(patched_ann_root):
        os.mkdir(patched_ann_root)
    
    ann_path = os.path.join(ann_root, img_id+'.txt')
    anns = open(ann_path, 'r').readlines()
    new_ann_path = os.path.join(patched_ann_root, img_id+'.txt')
    new_anns_file = open(new_ann_path, 'w')
    
    # calculate coordinates of four vertices of rotate bounding box
    patch[~mask] = 0
    points = cal_patch_poly(patch.squeeze(0) * 255)
    
    # add stick point offset
    points[:, 0] += point[1]
    points[:, 1] += point[0]
    
    # make annotation format 
    points_list = points.reshape(-1).tolist()
    points_str = [str(i) for i in points_list]
    category = obj_infos['category']
    new_ann = [img_id+'.jpg', '1', category, '0 0 0 0']
    new_ann = new_ann + points_str
    str_new_ann = ' '.join(new_ann) + '\n'
    anns.append(str_new_ann)

    # write to file
    for ann in anns:
        new_anns_file.write(ann)
    new_anns_file.close()

    ## check the anno is true
    img_save_path = './results'
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    img = cv2.imread(new_img_path)
    image = cv2.drawContours(img, [points], 0, (0, 0, 255), 2)
    save_img_name = img_id.split('.')[0] + '_' + name + '_rec_patch.jpg'
    cv2.imwrite(img_save_path + '/' + save_img_name, image)

    ## Step (6): save eval annotation
    print('------------------Part (4): Save eval annotation.------------------') 
    eval_ann_root = '/media/datasets/rotate_labels_buaa/train/annotations_eval/'
    if not os.path.exists(eval_ann_root):
        os.mkdir(eval_ann_root)

    eval_ann_path = os.path.join(eval_ann_root, img_id+'.txt')
    eval_anns_file = open(eval_ann_path, 'w')

    eval_ann = []
    eval_ann.append(category)
    eval_ann.append(name)
    eval_ann.extend([str(i) for i in rotate_param])
    eval_ann.extend([str(i) for i in point])
    eval_ann.extend([str(i) for i in patch_size])

    eval_anns_file.write(' '.join(eval_ann) + '\n')
    
    
