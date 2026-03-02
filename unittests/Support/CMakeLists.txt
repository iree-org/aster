add_aster_unittest(ASTERGraphTests
  GraphTest.cpp
)

target_link_libraries(ASTERGraphTests PRIVATE
  ASTERSupport
)
