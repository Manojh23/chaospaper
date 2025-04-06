%======================================================================
% Script Name: classify_all_teams.m
%
% Description:
%   This script reads all *.csv files from the specified folder.
%   For each CSV, it extracts the columns in 'columns_to_process' and
%   classifies them using 'chaos_modified'. If the data is stationary and
%   wavelet denoising is used, we store the denoised column in a new table.
%   Finally, we write that denoised table to a CSV in:
%       E:\chaospaper\denoisedcsv
%
%   The classification results include:
%       - chaos classification
%       - (if available) K-statistic
%       - (if available) permutation entropy
%
%   Results are appended to a text file named [team_name, '_results.txt'].
%======================================================================

clear; clc;

% --- 1) Setup the folder and file discovery ---
folder_path = "E:\chaosgrandfinale\teams2_perspective";
folder_path1 = 'E:\chaosgrandfinale\correctdenoising';
csv_files = dir(fullfile(folder_path, '*.csv'));

% Define the folder where you want to store denoised CSVs
denoised_folder = 'E:\chaosgrandfinale\denoising2';

% Create output folders if they don't exist
if ~exist(folder_path1, 'dir')
    mkdir(folder_path1);
end
if ~exist(denoised_folder, 'dir')
    mkdir(denoised_folder);
end

% --- 2) Define the columns you want to process ---
columns_to_process = {"FTGoalsFor", "FTGoalsAgainst", "TeamGS", "TeamGC", "TeamPoints", "MatchWeek", ...
 "TeamFormPts", "WinStreak3", "WinStreak5", "LossStreak3", "LossStreak5", ...
 "TeamGD", "TeamDiffPts", "TeamDiffFormPts"};

% --- 3) Loop through all CSV files in the folder ---
for i = 1:length(csv_files)
    % Get the CSV filename and full path
    csv_name = csv_files(i).name;
    full_csv_path = fullfile(folder_path, csv_name);

    % Read the CSV into a table
    T = readtable(full_csv_path, 'PreserveVariableNames', true);

    % Extract a base name for the text file (e.g. remove extension)
    [~, team_name, ~] = fileparts(csv_name);

    % Construct the output text filename for storing chaos classification
    results_filename = fullfile(folder_path1, [team_name, '_results.txt']);

    % Open the output text file in append mode ('a')
    fid = fopen(results_filename, 'a');
    if fid == -1
        error('Cannot open file: %s', results_filename);
    end

    fprintf(fid, '\n==============================================================\n');
    fprintf(fid, 'Chaos Classification Results for %s\n', team_name);
    fprintf(fid, 'Date/Time: %s\n', datestr(now));
    fprintf(fid, '==============================================================\n');

    % Prepare a table to store denoised columns (if stationary + wavelet used)
    denoised_T = table();

    % --- 4) Process each column of interest ---
    for c = 1:numel(columns_to_process)
        col_name = columns_to_process{c};

        % Check if the column exists in the table
        if ~ismember(col_name, T.Properties.VariableNames)
            % Column not found
            fprintf(fid, 'Column: %s NOT FOUND in the dataset.\n', col_name);
            fprintf(fid, '--------------------------------------------------------------\n');
            continue;
        end

        % Extract the data for that column
        y = T.(col_name);

        % Add a debug statement in the command window (optional)
        fprintf('Processing %s - Column: %s\n', team_name, col_name);

        % -- 4a) Wrap chaos_modified call in try/catch to detect errors --
        try
            % Run chaos_modified with the desired parameters
            output = chaos_modified( ...
                y, ...                 % Data
                [], ...                % cutoff (empty => auto)
                'adf', ...            % stationarity_test
                'schreiber', ...      % denoising_algorithm
                0, ...                % gaussian_transform
                'AAFT', ...       % surrogate_algorithm
                'downsample', ...     % downsampling_method
                0.5 ...               % sigma
            );

            % --- 5) Print results to the text file ---
            fprintf(fid, 'Column: %s\n', col_name);
            fprintf(fid, '  Classification: %s\n', output.result);


            
            if isfield(output, 'stochastic') && ~isempty(output.stochastic)
                fprintf(fid, '  stochastic: %d\n', output.stochastic);
            end
            

            % Optionally print the K-statistic and permutation entropy
            if isfield(output, 'K') && ~isempty(output.K)
                fprintf(fid, '  K-statistic: %.4f\n', output.K);
            end
            if isfield(output, 'permutation_entropy') && ~isempty(output.permutation_entropy)
                fprintf(fid, '  Permutation Entropy: %.4f\n', output.permutation_entropy);
            end
            fprintf(fid, '--------------------------------------------------------------\n');

            % --- 6) If data is stationary and we used wavelet or schreiber etc., store the denoised data
            %         (In your code, you used 'wavelet' sometimes; here you used 'schreiber'.
            %          If you want to store data after 'schreiber', that’s fine too.)
            if ~strcmpi(output.result, 'nonstationary') && ~isempty(output.denoised_data)
                denoised_T.(col_name) = output.denoised_data(:);
            end

        catch ME
            % If chaos_modified or surrogate code fails, catch the error
            fprintf(fid, 'ERROR encountered for column: %s\n', col_name);
            fprintf(fid, '  Error message: %s\n', ME.message);
            fprintf(fid, 'Skipping column: %s\n', col_name);
            fprintf(fid, '--------------------------------------------------------------\n');

            % Optionally print debug info in the command window
            fprintf('*** ERROR in %s, column %s ***\n', team_name, col_name);
            disp(ME.message);

            % Skip further processing for this column
            continue;
        end
    end

    % Close the file handle
    fclose(fid);

    % --- 7) Write out the denoised data (if any) to a CSV in the denoised folder
    denoised_filename = fullfile(denoised_folder, [team_name, '_denoised.csv']);
    writetable(denoised_T, denoised_filename);

    % Display a message in the MATLAB command window
    fprintf('Processed team: %s. Denoised columns saved to %s\n', team_name, denoised_filename);
end

fprintf('\nAll teams processed successfully.\n');
