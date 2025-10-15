import torch
import torch.nn.functional as F

def batch_compute_diff_3d(img):
    img_padded = F.pad(img, (1, 1, 1, 1, 1, 1), mode='constant', value=0).to(img.device).to(img.dtype)
    diff_right = (img - img_padded[:, :, 1:-1, 2:  , 1:-1]).abs()
    diff_left  = (img - img_padded[:, :, 1:-1,  :-2, 1:-1]).abs()
    diff_down  = (img - img_padded[:, :, 1:-1, 1:-1, 2:  ]).abs()
    diff_up    = (img - img_padded[:, :, 1:-1, 1:-1,  :-2]).abs()
    diff_front = (img - img_padded[:, :, 2:  , 1:-1, 1:-1]).abs()
    diff_back  = (img - img_padded[:, :,  :-2, 1:-1, 1:-1]).abs()

    # Concatenate differences along the channel dimension
    output = torch.cat([
        diff_right,
        diff_left,
        diff_down,
        diff_up,
        diff_front,
        diff_back
    ], dim=1)

    return output[
        :,:,
        1:-1,
        1:-1,
        1:-1,
    ]

def batch_compute_diff_2d(img):
    # f, c, h, w
    def compute_diff(img):
        c, h, w = img.shape
        diffs = []
        img_padded = F.pad(img, (1, 1, 1, 1), mode='constant', value=0)
        for i, (dx, dy) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]): # right, left, down, up
            diff = (img - img_padded[:, 1 + dx:h + 1 + dx, 1 + dy:w + 1 + dy]).abs()
            diffs.append(diff[:,1:-1,1:-1])
        output = torch.cat(diffs, dim=0)
        return output
    return torch.cat([
        compute_diff(img[b])[None,:] for b in range(img.shape[0])
    ],dim=0)

if __name__ == '__main__':
    input_tensor = torch.randn(2, 3, 4, 5, 6)
    output_tensor = batch_compute_diff_3d(input_tensor)
    print(output_tensor.shape)  # Should be (2, 18, 4, 5, 6)
