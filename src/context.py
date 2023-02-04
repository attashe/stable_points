

class Context:
    # Settings
    changed = False

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
    fill = 'default'
    use_inpaint_model = True
    
    ## Depth models settings
    adabins_weights = "weights/adabit.pth"
    midas_weights = "weights/mida.pth"
    near = 0.1
    far = 100.0
    rescale_depth = True
    depthscale = 3.0
    perspective_func = 'l1'  # 'l1' or 'l2'
    
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
    
    focal_length = 2.0
    
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