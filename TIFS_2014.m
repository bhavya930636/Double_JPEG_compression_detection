function [trunc_block, round_block, dct_error_images, error_images, final_feat] = TIFS_2014(image_path)
    final_feat = [];
    error_images = cell(length(image_path), 1);  % Use cells instead of arrays
    dct_error_images = cell(length(image_path), 1);  % Use cells instead of arrays
    trunc_block = [];
    round_block = [];
    
    for img = 1:length(image_path)  % Changed parfor to for to troubleshoot
        try
            trun = 0;
            img_path = image_path{img,1};
            
            % Print image path and size
            I = imread(img_path);
            fprintf('Processing image %d: %s, Size: %dx%d\n', img, img_path, size(I,1), size(I,2));
            
            jpeg_img = jpeg_read(img_path);
            rec = jpeg_rec_gray(jpeg_img);
            
            % Ensure both images have the same size
            [h, w] = size(I);
            h_new = floor(h/8) * 8;
            w_new = floor(w/8) * 8;
            
            I = I(1:h_new, 1:w_new);
            rec = rec(1:h_new, 1:w_new);
            
            Q = jpeg_img.quant_tables{1,1};
            Q_rep = repmat(Q, h_new/8, w_new/8);
            
            R = double(I) - rec;
            M = int64(bdct(R)./Q_rep);
            err = R;
            dct_err = bdct(err);
            
            % Store in cell arrays instead of concatenating
            err_reshaped = reshape(err, [1, 1, size(err,1), size(err,2)]);
            dct_err_reshaped = reshape(dct_err, [1, 1, size(dct_err,1), size(dct_err,2)]);
            
            error_images{img} = err_reshaped;
            dct_error_images{img} = dct_err_reshaped;
            
            zero_8 = zeros(8,8);
            
            r_error = [];
            r_error_dc = [];
            r_error_ac = [];
            t_error = [];
            t_error_dc = [];
            t_error_ac = [];

            for i = 1:8:size(M,1)
                for j = 1:8:size(M,2)
                    if i+7 <= size(M,1) && j+7 <= size(M,2)  % Boundary check
                        M_n = M(i:i+7, j:j+7);
                        if (nnz(M_n == zero_8) ~= 64)
                            R_n = R(i:i+7, j:j+7);
                            W_n = double(M_n).*Q;
                            W_n = reshape(W_n, 1, 64);
                            
                            if(max(max(R_n)) <= 0.5 && min(min(R_n)) >= -0.5)
                                trun = 0;
                                r_error = [r_error, R_n];
                                r_error_dc = [r_error_dc, W_n(1)];
                                r_error_ac = [r_error_ac, W_n(2:end)];
                            else
                                trun = 1;
                                t_error = [t_error, R_n];
                                t_error_dc = [t_error_dc, W_n(1)];
                                t_error_ac = [t_error_ac, W_n(2:end)];
                            end
                        end
                    end
                end
            end
      
            if(trun == 0)
                trunc_block = [trunc_block, 0];
                round_block = [round_block, 1];
            else
                trunc_block = [trunc_block, 1];
                round_block = [round_block, 0];
            end
            
            % Handle empty arrays for feature calculation
            if isempty(r_error)
                r_error_mean = 0;
                r_error_var = 0;
            else
                r_error_mean = mean(abs(r_error(:)));
                r_error_var = var(abs(r_error(:)));
            end
            
            if isempty(t_error)
                t_error_mean = 0;
                t_error_var = 0;
            else
                t_error_mean = mean(abs(t_error(:)));
                t_error_var = var(abs(t_error(:)));
            end
            
            % Similarly handle other feature calculations
            if isempty(r_error_dc)
                r_error_dc_mean = 0;
                r_error_dc_var = 0;
            else
                r_error_dc_mean = mean(abs(r_error_dc));
                r_error_dc_var = var(abs(r_error_dc));
            end
            
            if isempty(r_error_ac)
                r_error_ac_mean = 0;
                r_error_ac_var = 0;
            else
                r_error_ac_mean = mean(abs(r_error_ac));
                r_error_ac_var = var(abs(r_error_ac));
            end
            
            if isempty(t_error_dc)
                t_error_dc_mean = 0;
                t_error_dc_var = 0;
            else
                t_error_dc_mean = mean(abs(t_error_dc));
                t_error_dc_var = var(abs(t_error_dc));
            end
            
            if isempty(t_error_ac)
                t_error_ac_mean = 0;
                t_error_ac_var = 0;
            else
                t_error_ac_mean = mean(abs(t_error_ac));
                t_error_ac_var = var(abs(t_error_ac));
            end
            
            % Calculate ratio with checks to avoid division by zero
            if isempty(r_error_dc) && isempty(t_error_dc)
                ratio = 0;
            elseif isempty(t_error_dc)
                ratio = 1;
            else
                ratio = length(r_error_dc) / (length(r_error_dc) + length(t_error_dc));
            end
            
            feature_vec = [r_error_mean, r_error_var, ...
                          t_error_mean, t_error_var, ...
                          r_error_dc_mean, r_error_dc_var, ...
                          r_error_ac_mean, r_error_ac_var, ...
                          t_error_dc_mean, t_error_dc_var, ...
                          t_error_ac_mean, t_error_ac_var, ...
                          ratio];
                          
            final_feat = [final_feat; feature_vec];
            
        catch e
            fprintf('Error processing image %d: %s\n', img, e.message);
            fprintf('Stack: %s\n', e.getReport);
            
            % Add placeholder data for this failed image
            trunc_block = [trunc_block, 0];
            round_block = [round_block, 0];
            error_images{img} = [];
            dct_error_images{img} = [];
            final_feat = [final_feat; zeros(1, 13)];
        end
    end
    
    % Convert cell arrays back to matrices if needed for your application
    % This would need custom handling based on how you use these outputs
end