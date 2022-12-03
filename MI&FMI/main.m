%% Reference: Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018. 
%% https://arxiv.org/abs/1804.08361

MIS = zeros(20, 1);
FMI = zeros(20, 1);
for i=1:20
    fileName_source_ct  = dir("images/ct_*.jpg");
    fileName_source_mri = dir("images/mri_*.jpg");
    fileName_fused      = dir("images/fused_*.jpg");
    
    source_image1 = imread(strcat("images/", fileName_source_ct(i).name));
    source_image2 = imread(strcat("images/", fileName_source_mri(i).name));
    fused_image   = imread(strcat("images/", fileName_fused(i).name));
    
    disp("Start");
    disp('---------------------------Analysis---------------------------');
    %%[EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(fused_image,source_image1,source_image2);
    MI = analysis_MI(source_image1,source_image2,fused_image);
    FMI_pixel = analysis_fmi(source_image1,source_image2,fused_image);
    disp('One pairDone');
    MIS(i) = MI;
    FMI(i) = FMI_pixel;
end

