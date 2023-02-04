def max_pulling_callback(sender, app_data):
    # Context.mask = max_pulling(Context.mask)
    mask = Context.mask
    image = Context.rendered_image
    
    im_floodfill = mask.copy()

    h, w = img.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    
    ret, imf, maskf, rect = cv2.floodFill(im_floodfill, flood_mask, (0,0), 255)
    maskf = maskf[1:-1, 1:-1]
    
    img_conv = torch.tensor(image, dtype=torch.float32)
    img_conv = torch.permute(img_conv, (2, 0, 1)).unsqueeze(0)

    img_max = F.max_pool2d(img_conv, 3, 1, 1)
    img_max_numpy = img_max.permute(0, 2, 3, 1).numpy()[0]
    img_max_numpy = img_max_numpy.astype(np.uint8)
    
    mask_conv = 255 - mask
    mask_conv = mask_conv.reshape(1, 1, h, w)

    mask_max = F.max_pool2d(torch.tensor(mask_conv.astype(np.float32)), 3, 1, 1)
    mask_max_numpy = mask_max.numpy().astype(np.uint8)[0][0]
    mask_max_numpy = 255 - mask_max_numpy
    
    img_max_numpy[maskf == 1] = 0
    mask_max_numpy[maskf == 1] = 255
    
    np.copyto(Context.rendered_image, img_max_numpy)
    np.copyto(Context.mask, mask_max_numpy)
    
    np.copyto(Context.mask_data[..., 0], Context.mask.astype(np.float32) / 255)
    np.copyto(Context.texture_data, Context.rendered_image.astype(np.float32) / 255)