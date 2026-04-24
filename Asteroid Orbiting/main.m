%% The Main Entry Point Into Asteroid Orbiter
% Essentially this establishes the CLI tool
clc;

fprintf("Loading Kernel Data From kernel_data.mat\n");
load("kernel_data.mat", "path_to_mice", "path_to_generic_kernels");

fprintf("Generic Kernel Path: %s\n Mice Path: %s\n", ...
    path_to_generic_kernels, path_to_mice);

usr_in = input("Is this correct? (y/n):", 's');

if usr_in ~= 'y' && usr_in ~= 'Y'
    path_to_generic_kernels = convertStringsToChars(input("Paste the Generic Kernel Path:", 's'));
    path_to_mice = convertStringsToChars(input("Paste the MICE Path:", 's'));
end

save("kernel_data.mat", "path_to_mice", "path_to_generic_kernels")

load_kernels("bsp_files", path_to_mice, path_to_generic_kernels);

fprintf("Kernel data loaded successfully.\n");


fprintf("Cleaning up...\n");
cspice_kclear;
fprintf("Cleanup successful.\n");