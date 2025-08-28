import inspect
import ecletp.resqml_builder as rb
import ecletp.main as m

print("build_in_memory signature:", inspect.signature(rb.build_in_memory))
print("build_in_memory defined at:", rb.build_in_memory.__code__.co_filename)

print("main.build_and_??? defined at:", m.__file__)
# show the line with the call to build_in_memory
import linecache, re
src = open(m.__file__, 'r').read().splitlines()
for i, line in enumerate(src, start=1):
    if re.search(r'build_in_memory\\(', line):
        print(f"Line {i}: {line.strip()}")

