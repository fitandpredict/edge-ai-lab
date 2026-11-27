# Packs every .espdl in models/ (our own espdet_pico models) into flash rodata, like coco_detect does.
# After adding or renaming a model run `idf.py reconfigure`: the list is read at configure time.
idf_build_get_property(component_targets __COMPONENT_TARGETS)
idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

file(GLOB models ${CMAKE_CURRENT_LIST_DIR}/models/*.espdl)
list(SORT models)
set(names "")
foreach(m ${models})
    get_filename_component(n ${m} NAME)
    list(APPEND names ${n})
endforeach()
string(REPLACE ";" "," names_csv "${names}")

idf_component_register(SRCS "espdet_models.cpp" INCLUDE_DIRS "." REQUIRES esp-dl)
target_compile_definitions(${COMPONENT_LIB} PRIVATE ESPDET_MODEL_NAMES="${names_csv}")

set(packed_model ${BUILD_DIR}/espdl_models/espdet_models.espdl)
file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)
add_custom_command(
    OUTPUT ${packed_model}
    COMMENT "Pack espdet models: ${names_csv}"
    COMMAND python ${espdl_dir}/fbs_loader/pack_espdl_models.py --model_path ${models} --out_file ${packed_model}
    DEPENDS ${models}
    VERBATIM)
target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
