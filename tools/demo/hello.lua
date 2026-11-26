-- ringzero demo (M8): `!run lua /demo/hello.lua`
local words = {}
for w in ("the quick brown fox jumps over the lazy dog the end"):gmatch("%a+") do
  words[w] = (words[w] or 0) + 1
end
local keys = {}
for k in pairs(words) do keys[#keys + 1] = k end
table.sort(keys, function(a, b) return words[a] > words[b] or (words[a] == words[b] and a < b) end)
for i = 1, 3 do print(keys[i], words[keys[i]]) end

local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
local t = os.clock()
print(("fib(20) = %d in %.0f ms"):format(fib(20), (os.clock() - t) * 1000))

-- errors are caught by pcall (setjmp/longjmp in the C code)
local ok, err = pcall(function() error("something went wrong") end)
print("pcall:", ok, err)

local f = assert(io.open("/demo/lua-out.txt", "w"))
f:write("written by ", _VERSION, "\n")
f:close()
print(io.open("/demo/lua-out.txt"):read("a"))
