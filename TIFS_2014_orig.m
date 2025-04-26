% This file obtains the error images from the image_paths and saves it in the same location. 
function [trunc_block, round_block,dct_error_images, error_images, final_feat] = TIFS_2014(image_path) %trunc_block: 1 if image has truncation blocks, 0 otherwise.round_block: 1 if image has only rounding error blocks, 0 otherwise.dct_error_images: DCT domain reconstruction error for each image.error_images: spatial domain reconstruction error for each image.final_feat: statistical features extracted from DCT blocks (used for classification).
      final_feat = [];
      error_images = [];
      dct_error_images = [];
      trunc_block = [];
      round_block = [];
      parfor img = 1:length(image_path)
            trun = 0;
            img_path = image_path{img,1};
            jpeg_img = jpeg_read(img_path);%Reads JPEG compression metadata (quantization table, DCT coeffs).
            I = imread(img_path);%Reads the image as a matrix.
            fprintf('Processing image %d: %s, Size: %dx%d\n', img, img_path, size(I,1), size(I,2));
            
            rec = jpeg_rec_gray(jpeg_img);%Reconstructs grayscale image using JPEG DCT blocks.
            Q = jpeg_img.quant_tables{1,1};
            Q_rep = repmat(Q, size(I,1)/8, size(I,2)/8);
            
            R = double(I) - rec;%Spatial-domain reconstruction error image.
            M = int64(bdct(R)./Q_rep);%DCT error quantized by the JPEG quantization table → highlights unstable blocks
            err = R;
            dct_err = bdct(err);
            err = reshape(err,[1,1,size(err,1),size(err,2)]);
            dct_err = reshape(dct_err,[1,1,size(dct_err,1),size(dct_err,2)]);
            zero_8 = zeros(8,8); % For checking stability
            
            r_error = [];%DC component: the first DCT coefficient (average brightness)
%AC components: the rest (fine details, edges)
            r_error_dc = [];
            r_error_ac = [];
            t_error = [];
            t_error_dc = [];
            t_error_ac = [];

            for i = 1:8:size(M,1) %Iterates through each 8×8 block
                for j = 1:8:size(M,2)
                    M_n = M(i:i+7, j:j+7);
                    % Process unstable block only
                    if (nnz(M_n == zero_8) ~= 64) %If block is not all zero after quantization (i.e., unstable), process it.
                        R_n = R(i:i+7, j:j+7);
                        W_n = double(M_n).*Q; % Dequantized DCT of the error block
                        W_n = reshape(W_n, 1, 64);
%If the error is in [−0.5,0.5] it's rounding error.Otherwise, it's truncation error.
                        % Rounding error block
                        if(max(max(R_n)) <= 0.5 && min(min(R_n)) >= -0.5)
                            trun = 0;
                            r_error = [r_error, R_n];
                            r_error_dc = [r_error_dc, W_n(1)]; % DC comp
                            r_error_ac = [r_error_ac, W_n(2:end)]; % AC comp

                        % Truncation error block
                        else
                            trun = 1;
                            t_error = [t_error, R_n];
                            t_error_dc = [t_error_dc, W_n(1)]; % DC comp
                            t_error_ac = [t_error_ac, W_n(2:end)]; % AC comp
                        end            
                    end
                end
            end
      
            if(trun == 0)
                trunc_block = [trunc_block,0];
                round_block = [round_block,1];
            else
                trunc_block = [trunc_block,1];
                round_block = [round_block,0];
            end
            % Make the final feature vector
            feature_vec = [mean(abs(r_error(:))), var(abs(r_error(:))),mean(abs(t_error(:))), var(abs(t_error(:))),
    mean(abs(r_error_dc)), var(abs(r_error_dc)),
    mean(abs(r_error_ac)), var(abs(r_error_ac)),
    mean(abs(t_error_dc)), var(abs(t_error_dc)),
    mean(abs(t_error_ac)), var(abs(t_error_ac)),
    length(r_error_dc) / (length(r_error_dc) + length(t_error_dc))
];
%This gives 13 features per image:Mean & variance of spatial rounding error Mean & variance of spatial truncation error Mean & variance of DC rounding error Mean & variance of AC rounding error Mean & variance of DC truncation error Mean & variance of AC truncation error Ratio of rounding DC blocks to total
            final_feat = [final_feat;feature_vec];
            error_images = [error_images; err];
            dct_error_images  = [dct_error_images;dct_err];
      end
end

       
    
 
    
