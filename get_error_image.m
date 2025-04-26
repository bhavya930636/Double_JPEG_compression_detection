function error = get_error_image(image_path)
  folder = image_path.folder;
  error = [];
  parfor h = 1:length(image_path)
    if(rem(h,1000)==0) %Every 1000 images, it prints the current index to show progress.
        h
    end
    read_path = strcat(folder,'/', image_path(h).name);
    jpeg_1 = jpeg_read(read_path); %Uses jpeg_read from JPEG Toolbox to get JPEG compression structure (quantization tables, DCT coefficients, etc.).
    I1  = imread(read_path);%Reads the original image into matrix form (standard MATLAB image array).
    rec1 = jpeg_rec_gray(jpeg_1); %Reconstructs the grayscale version of the image from the JPEG DCT coefficients.
    err = double(I1) - rec1; %error=Original Image−JPEG-Reconstructed Grayscale
    err = reshape(err,[1,1,size(err,1),size(err,2)]);%Reshapes the error matrix into 4D: [1,1,H,W]. This likely prepares data for batch processing in deep learning (common in CNNs).
    error = [error;err];%Appends the reshaped error to the final output array.
  end
end
%parfor is a parallel for loop in MATLAB to speed up the loop using multiple CPU cores. Iterates over all images.
    
