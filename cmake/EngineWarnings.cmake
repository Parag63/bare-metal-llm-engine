# EngineWarnings.cmake
# One place to define the warning set, applied to every target.

function(engine_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    if(ENGINE_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>)
    if(ENGINE_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
  endif()
endfunction()
