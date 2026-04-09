function asteroid_list = LOAD_KERNELS(BSP_FOLDER_NAME)
    path_to_generic_kernels = 'C:\Users\SPLog\OneDrive\Documents\Ae105A\kernels';
    path_to_mice            = 'C:\Users\SPLog\OneDrive\Documents\Ae105A\mice';
    addpath([path_to_mice,'\src\mice']);
    addpath([path_to_mice,'\lib']);

    cspice_furnsh( [path_to_generic_kernels,'\lsk\naif0012.tls.pc']);
    cspice_furnsh( [path_to_generic_kernels,'\spk\satellites\jup310.bsp']);
    cspice_furnsh( [path_to_generic_kernels,'\spk\planets\de430.bsp']);
    cspice_furnsh( [path_to_generic_kernels,'\pck\gm_de431.tpc']);
    cspice_furnsh( [path_to_generic_kernels,'\pck\pck00010.tpc']);

    bsp_files = dir(join([BSP_FOLDER_NAME, "\*.bsp"]));

    asteroid_list(length(bsp_files)) = struct('ID', [], 'NAME', []);

    for i = 1:length(bsp_files)
        formatted_bsp_file = char(join([BSP_FOLDER_NAME, "\", bsp_files(i).name], ""));
        cspice_furnsh(formatted_bsp_file);

        % 3. Get the IDs from the SPK
        [id] = cspice_spkobj( formatted_bsp_file, 1000 );
        asteroid_list(i).ID = id;
        asteroid_list(i).NAME = bsp_files(i).name(1:end-4);
    end

end