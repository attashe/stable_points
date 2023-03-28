

class Context:
    # Settings
    changed = False
    image_path = ''

    ## Saving settings
    results_folder = 'output'
    basename = 'run_'
    log_folder = 'run'
    save_idx = 0
    render_image_idx = 0
    
    # Default image size
    image_width = 512
    image_height = 512
    
    ## Stable diffusion weights path
    sd_weights = "weights/stable_diffusion.pth"
    sd_inpaint_weights = "weights/stable_diffusion_inpaint.pth"
    
    ## Generation  txt2img settings
    prompt = 'a painting of a nerdy rodent'
    num_samples = 1
    seed = 42
    scale = 9.0
    steps = 20
    sampler = 'ddim'
    ddim_eta = 0.0
    
    ## Inpainting settings
    inpainter = None
    fill = 'default'
    use_inpaint_model = True
    use_automatic_api = False
    use_controlnet = False
    api = None
    api_model_name = 'sd-v1-5-inpainting'
    transform_inpaint = False
    
    # Remove alone points
    points_thresh = 3
    points_radius = 0.5
    
    ## Depth models settings
    use_depth_model = True
    depth_type = 'midas'  # Default depth model type 'midas' | 'adabins' | 'leres'
    depth_model = None
    depth_resolution = 384
    midas_resolution_default = 384
    adabins_resolution_default = 448
    leres_resolution_default = 448
    adabins_weights = "weights/adabit.pth"
    midas_weights = "weights/mida.pth"
    depthscale = 15.0  # 3.0, 15.0
    depth_gamma = 1.0
    depth_alpha = 0.1
    near = 15.0
    far = 100.0
    rescale_depth = True
    perspective_func = 'l1'  # 'l1' or 'l2'
    use_depthmap_instead_mask = False
    depth_thresh = 0.5
    
    ## Camera settings
    alpha_step = 0.01
    beta_step = 0.01
    theta_step = 0.01
    radius_step = 0.01
    
    default_camera = {
        "pos_x": 0.0,
        "pos_y": 0.0,
        "pos_z": -1.0,
        "rot_x": 0.0,
        "rot_y": 0.0,
        "rot_z": 0.0,
    }
    
    camera_mode = 'arcball'
    control_mode = 'rotate'
    
    focal_length = 1.0
    
    x_step = 3.0
    y_step = 3.0
    translation_step = 0.5
    
    ## Prerender rescale
    upscale = 2
    downscale = 3
    
    # Global variables
    rendered_image = None
    rendered_depth = None
    inpainted_image = None
    
    mask = None