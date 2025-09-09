% Load 4th order RIRs, normalises to -40 dBFS (according to the omni
% channel) and saves to another directory

read_dir = "Audio/Asymmetry Unnorm/";
write_dir = "Audio/Asymmetry/";
output_bit_depth = 32;

files = dir(fullfile(read_dir,"*.wav"));

loudness_target_dB_LUFS = -40;

[~, fs] = audioread(read_dir + files(1).name);

for file_index = 1:numel(files)
    [ir, ~] = audioread(read_dir + files(file_index).name);

    output_filename = write_dir + file_index + ".wav";

    if isfile(output_filename)
        disp("File already exists; skipping...")
        continue
    end

    % Normalise output based on the omnidirectional channel
    loudness_dB_LUFS = integratedLoudness(ir(:, 1), fs);
    gain_to_apply_dB = loudness_target_dB_LUFS - loudness_dB_LUFS;

    ir = ir * power(10.0, gain_to_apply_dB / 20.0);

    audiowrite(output_filename, ir, fs, "BitsPerSample", output_bit_depth);
end

disp("Stimulus Generation Complete");