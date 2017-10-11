file(REMOVE_RECURSE
  "../lib/libShapeMatch.pdb"
  "../lib/libShapeMatch.dylib"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/ShapeMatch.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
