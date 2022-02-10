
if (HAVE_CXX_RESTRICT)
  set (KEYWORD_RESTRICT "restrict")
elif (HAVE_CXX___RESTRICT)
  set (KEYWORD_RESTRICT "__restrict")
else ()
  set (KEYWORD_RESTRICT "")
endif ()


add_subdirectory(idioms)
#add_subdirectory(qmcpack)
