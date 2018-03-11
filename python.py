#1
'''
Python的数据模型
'''

#2
'''
容器序列
list tuple collections.deque等
扁平序列
str bytes bytearray memoryview array.array
扁平序列保存的是值
而容器序列保存的是对象的引用
'''

'''
可变序列
list bytearray array.array collections.deque memoryview
不可变序列
tuple str bytes
'''

'''
元祖的拆包
'''
tup = (1,2,3)
a, b, c = tup

#用*处理剩下的元素
a, b, c, *rest = range(9)


'''
具名元祖
'''
Card = collections.namedtuple('Card', ['rank', 'suit'])
a = Card(rank='b', suit='c')

'''
切片
实现切片的功能需要正确实现__setitem__和__getitem__将在后面讲解
'''

#3
'''
尽量使用isinstance(obj, 基类)判断类型
'''
if isinstance(obj, collections.Mapping)

#4

 
'''当使用闭包要修改不可变的对象时，使用nolocal把对象声明为自由变量存储在object.__closure__中'''

'''函数运行所用时间的装饰器'''
import time
import functools
def clock(func):
	@functools.wraps(func)
	def clocked(*args,**kwargs):
		t0 = time.perf_counter()
		result = func(*args)
		elapsed = time.perf_counter() - t0
		name = func.__name__
		arg_lst = []
		if args:
			arg_lst.append(', '.join(repr(arg) for arg in args))
		if kwargs:
			pairs = ['%s=%r' %(k,w) for k,w in sorted(kwargs.items())]
			arg_lst.append(', '.join(pairs))
		arg_str = ', '.join(arg_lst)
		print('[%0.8fs] %s(%s) -> %r' %(elapsed,name,arg_str,result))
		return result
	return clocked

#第八章
'''
浅复制最简单的方法是使用内置的类型构造方法
对于列表和其他可变序列来说可以是用切片[:]创建副本
也可以使用copy模块的copy函数
'''
import copy
l1 = [1,2,3,4,5,6]
l2 = list(l1)
l3 = l1[:]
l4 = copy.copy(l3)

'''
深复制使用copy模块的deepcopy函数
'''
l5 = copy.deepcopy(l1)

'''
Python唯一支持的参数传递是共享传递，多数面向对象语言
都是使用该模式
'''

'''
不要使用可变类型作为参数的默认值，用None作为替代
当要传入可变类型作为参数时要小心是否会对传入值进行
修改，必要时可以在传入后显式地复制副本
'''
class Class(object):
	def __init__(self, lst=None,):
		if lst is None:
			self.lst = []
		else:
			self.lst = list(lst)

'''
del语句删除的是变量而不是对象，CPython中垃圾回收
使用的主要算法是引用计数和分代垃圾回收算法
每个对象都会统计有多少个引用指向自己，当引用
计数为零是就是自动销毁对象
'''

'''
弱引用不会增加引用计数，在缓存应用中很有用
ref类是低层的接口，应该使用weakref模块中的
WeakValueDictionary,WeakKeyDictionary
WeakSet,finalize
'''
import weakref
a_set = {0,1,2,3,4}
#wref是a_set的弱引用，wref()获取对象
wref = weakref.ref(a_set)


#9

#python的各种内置方法(协议)
import math
import numbers
import reprlib
import functools
from array import array
from operator import xor
class Vector(object):

	typecode = 'd'

	#序列类型最好接受可迭代对象为参数
	def __init__(self, components):

		self._components = array(self.typecode, components)

	#返回一个可迭代对象
	def __iter__(self):

		return iter(self._components)

	def __repr__(self):

		cls_name = type(self).__name__
		components = reprlib.repr(self._components)
		components = components[components.find('['):-1]

		return '{}({})'.format(cls_name, components)

	def __str__(self):

		return str(tuple(self))

	def __bytes__(self):

		return (bytes(ord(self.typecode)) + 
					bytes(self._components))

	def __abs__(self):

		return math.sqrt(sum(x*x for x in self._components))

	def __bool__(self):

		return bool(abs(self))

	def __format__(self, fmt_spec=''):

		components = (format(value, fmt_spec) for value in self)
		return '({})'.format(', '.join(components))

	def __hash__(self):

		return functools.reduce(xor, (hash(i) for i in self), 0)

	def __int__(self):

		return [int(i) for i in self]

	def __float__(self):

		return [float(u) for i in self]

	def __len__(self):

		return len(self._components)

	def __getitem__(self, index):

		cls = type(self)
		if isinstance(index, slice):
			return cls(self._components[index])
		elif isinstance(index, numbers.Integral):
			return self._components[index]
		else:
			msg = '{cls.__name__} indices must be Integers'
			raise TypeError(msg.format(cls=cls))

	shortcut_name = 'xyz'

	def __getattr__(self, name):

		cls = type(self)
		if len(name) == 1:
			pos = shortcut_name.find(name)
			if 0 <= pos < len(self._components):
				return self._components[pos]

		msg = '{.__name__!r} object has no attribute {!r}'
		raise AttributeError(msg.format(cls, name))

	def __setattr__(self, name, value):

		cls = type(self)
		if len(name) == 1:
			if name in shortcut_name:
				error = 'readonly attribute {attr_name!r}'
			elif name.islower():
				error = "can't set attribute 'a' to 'z' in {cls_name!r}"
			else:
				error = ''

			if error:
				msg = error.format(attr_name=name, cls_name=cls.__name__)
				raise AttributeError(msg)

		super().__setattr__(name, value)

	#用classmethod来定义类的备用构造方法
	@classmethod
	def frombytes(cls, octets):

		typecode = chr(octets[0])
		memv = memoryview(octets[1:0]).cast(typecode)
		return cls(memv)


	#运算符重载要遵循返回新的实例不改变原有实例
	def __neg__(self):

		cls = type(self)
		return cls(-x for x in self)

	def __pos__(self):

		cls = type(self)
		return cls(self)

	#通过捕获异常来返回NotImplemented，更合理的方法在__mul__上实现
	#+ - * / // % divmod() **,pow() @ & | ^ << >>等运算符都适合这种重载方式
	def __add__(self, other):

		try:
			cls = type(self)
			pairs = itertools.zip_longest(self, other, fillvalue=0)
			return cls(a + b for a, b in pairs)

		except TypeError:
			return NotImplemented

	def __radd__(self, other):

		return self + other

	def __mul__(self, scalar):

		if isinstance(scalar, numbers.Real):
			cls = type(self)
			return cls(n*scalar for n in self)
		else:
			return NotImplemented

	def __rmul__(self, scalar):

		return self * scalar

	#比较运算符的处理和前面差不多
	#==正反都是调用__eq__
	#正向的__gt__方法调用的是方向的__lt__方法，并把参数对调
	#如果放回NotImplemented即调用另一个参数的相同方法
	#不用显式实现__ne__方法
	def __eq__(self, other):

		cls = type(self)
		if isinstance(other, cls):
			return len(self) == len(other) and all(a == b for a, b in zip(self, other))
		else:
			return NotImplemented

	#增量运算符+= *=等
	#不会修改不可变目标，而是新建实例重新绑定
	#用户定义的不可变类型不用实现就地的特殊方法
	def __iadd__(self, other):
		pass


#继承内置类型不会调用用户定义的特殊方法
class DoppelDict(dict):
	#使用内置类型的方法使会忽略用户定义的该方法
	def __setitem__(self, name, value):
		super().__setitem__(self, name, [value]*2)

'''
原生类型这种行为违背了面向对象编程的一个基本原则: 
始终应该从实例(self)所属的类搜索方法，正确的方法
是子类化python提供给用户的类: 例如collections的
UserDict等类型'''

import collections

class DoppelDict(collections.UserDict):
	def __setitem__(self, name, value):
		super().__setitem__(self, name, [value]*2)

'''多重继承中的方法解析顺序是按照类中一个名为__mro__
的属性(元祖)中的值的顺序来解析的，super函数就是遵守该属性
的顺序来引用超类'''
>>>D().__mro__
'(<class diamond.D>,<class diamond.B>,<class diamond.C>,<class diamond.A>)'

'''
检查对象是否为迭代器
'''
isinstance(obj, abc.Iterator)

'''
典型的迭代器设计模式
不应该混淆迭代器和可迭代对象
为了支持多种遍历，应该能从一个可迭代对象的实例
获取多个独立的迭代器而且各个迭代器能维护自身的内部状态
'''
import re
import reprlib

RE_WORD = re.compile('\w+')

class Sentence:

	def __init__(self, text):

		self.text = text
		self.words = RE_WORD.findall(text)

	def __repr__(self):

		return 'Sentence({})'.format(reprlib.repr(self.words))

	def __iter__(self):

		return SentenceIteator(self.words)

class SentenceIteator:

	def __init__(self, words):

		self.words = words
		self.index = 0

	def __next__(self):

		try:
			word = self.words[self.index]
		except IndexError:
			raise StopIteration()

		self.index += 1
		return word

	def __iter__(self):

		return self

'''
在Python中简化迭代器模式
'''
class Sentence:

	def __init__(self, text):

		self.text = text
		self.words = RE_WORD.findall(text)

	def __repr__(self):

		return 'Sentence({})'.format(reprlib.repr(self.words))

	def __iter__(self):

		#return iter(self.words)

		# for word in self.words:
		# 	yield word

		#惰性求值
		# for word in RE_WORD.finditer(self.text):
		# 	yield word.group()

		return (word.group() for word in RE_WORD.finditer(self.text))

'''
itertools模块有很多生成器函数可供使用
用于过滤 compress dropwhile filter filterfalse 
islice takewhile 用于映射 accumulate enumerate
map starmap 合并多个可迭代对象 chain chain.from_iterable
product zip zip_longest 
'''

'''
iter的特殊用法
'''
def rand():
	return randint(1, 6)

#当rand函数返回1时这个生成器才会停止
gen = iter(rand, 1)


'''
else在其他流程控制语句的使用
for:当for语句执行完成，没有被break语句终止时调用
while:当while语句条件为False而且没有被break语句终止时调用
try:当try语句没有抛出异常时调用
'''
for item in items:
	if item == 'banana':
		break
else:
	raise ValueError('No banana flavor found')

#为了流程更加清晰，else语句在try/except语句中很有用
try:
	dangerout_call()
except OSError:
	log('OSError...')
else:
	after_call()


'''
with语句的使用
'''
#what是__enter__的返回值
with LookingGlass() as what:

	print('Alice, Kitty and Snowdrop')
	print(what)

class LookingGlass(object):

	def __init__(self):
		pass

	def __enter__(self):

		import sys
		#把系统自带的标准输出函数保存
		self.original_write = sys.stdout.write
		#动态绑定系统的标准输出函数
		sys.stdout.write = self.reverse_write

		return 'JABBERWOCKY'

	def reverse_write(self, text):

		self.original_write(text[::-1])

	#当没有错误时参数全是None
	def __exit__(self, exc_type, exc_value, traceback):

		#重复导入不会消耗太多资源，Python会缓存导入的模块
		import sys
		sys.stdout.write = self.original_write

		if exc_type is ZeroDivisionError:
			print('Please DO NOT divide by zero!')
			return True

		#如果返回None或True以外的值，with块中的任何异常都会向上冒泡


'''
contextlib模块
'''
'''
@contextmanager的使用
'''
import contextlib

@contextlib.contextmanager
def looking_glass():

	import sys
	original_write = sys.stdout.write

	def reverse_write(text):

		original_write(text[::-1])

	sys.stdout.write = reverse_write

	msg = ''

	try:
		#yield前面作用与__enter__相同，后面与__exit__相同
		yield 'JABBERWOCKY'
	except ZeroDivisionError:
		msg = 'Please DO NOT divide by zero!'
	finally:
		sys.stdout.write = original_write
		if msg:
			print(msg)

with looking_glass() as what:

	print('Alice, Kitty and Snowdrop')
	print(what)



'''
协程
'''

def averager():

	count = 0
	total = 0

	while True:
		term = yield result
		total += term
		count += 1
		result = total/count



def averager():

	count = 0
	total = 0

	while True:
		term = yield
		if term is None:
			break
		total += term
		count += 1

	
	return Result(count, total/count)


'''
yield from的使用
使用该语句时有特定的结构
客户端-委派生成器-子生成器
'''
Result = collections.namedtuple('Result',['count', 'result'])

#子生成器
def averager():

	count = 0
	total = 0
	average = None
	while True:
		#在yield语句暂停
		#如果使用yield from语句的话，对委派生成器的操作(send)传入给该yield
		term = yield
		if term is None:
			break

		total += term
		count += 1

		average = total/count

	#return会抛出StopIteration异常，该值附着在异常的Value上
	#使用yield from语句的话该值为语句的返回值
	return Result(count, average)

#委派生成器
def grouper(results, key):

	while True:
		#每次循环创建新的协程
		#results[key]的值为averager return的值
		results[key] = yield from averager()


#客户端
def main(data):
#客户端接收data
	results = {}

	for key, values in data:
		#通过创建新的委派生成器来生成新的协程
		group = grouper(results, key)
		#激活委派生成器
		next(group)
		for value in values:
			#通过委派生成器发送数据到协程(子生成器)
			group.send(value)

		#发送None来停止子生成器
		group.send(None)

'''
协程的异常处理
协程中未处理的异常会向上冒泡，传给next函数或send方法的调用方
throw和close方法可以显式把异常发送给协程
使协程yield暂停的地方抛出指定异常

在使用yield from时的异常
传入委派生成器的异常(除了GeneratorExit)都会传给子生成器的throw方法
如果传入GeneratorExit或者调用close方法，异常会向上冒泡
'''



'''
用协程写出出租车仿真系统
此实例的要旨是说明如何在一个主循环中处理事件，
以及如何通过发送数据驱动协程，这是asynico包底层的基本思想
'''
Event = collections.namedtuple('Event',['time', 'ident', 'action'])

def texi_process(ident, trips, start_time=0):

	time = yield Event(start_time, ident, 'Leave Garage')

	for i in trips:
		time = yield Event(time, ident, 'Pick Up passenger')
		time = yield Event(time, indent, 'Drop Down passenger')

	yield Event(time, indent, 'going home')

from queue import PriorityQueue

DEPARTURE_INTERVAL = 5
num_taxis = 5
procs_map = {i: texi_process(i, (i+1)*2, i*DEPARTURE_INTERVAL) for i in range(num_taxis)}

class Simulator(object):
	
	def __init__(self, procs_map):

		self.events = PriorityQueue()
		self.procs = procs_map 

	def run(self, end_time):

		for _, proc in sorted(self.procs):
			first_event = next(proc)
			self.events.put(first_event)

		sim_time = 0
		while sim_time < end_time:
			if self.events.empty():
				print('*** end of events ***')
				break

			current_event = self.events.get()
			sim_time, proc_id, previous_action = current_event
			print('taxi:',proc_id,proc_id*' ',current_event)
			active_proc = self.procs[proc_id]
			next_time = sim_time + compute_duration(previous_action)
			try:
				next_event = active_proc.send(next_time)
			except StopIteration:
				del self.procs[proc_id]
			else:
				self.events.put(next_event)

		else:
			msg = '*** end of simulation time: {} events pending ***'
			print(msg.format(self.events.qsize()))

'''
concurrent.futures模块
ThreadPoolExecutor 
ProcessPoolExecutor是这个模块的主要特色
由于抽象层次很高，所以只能实现较简单的任务
若要实现较复杂的功能
需要使用threading和multiprocessing模块
'''
import time
from concurrent import futures

def main(download_many):

	t0 = time.time()
	count = download_many(cc_list)
	elapsed = time.time() - t0
	msg = '\n{} flags downloaded in {:.2f}s'
	print(msg.format(count, elapsed))

def get_flag(cc):

	url = ''
	resp = request.get(url)
	return resp.content

def download_one(cc):

	image = get_flag(cc)
	save_flag(image, cc.lower()+'.gif')
	return cc

def download_many(cc_list):

	workers = min(MAX_WORKERS, len(cc_list))
	#要换成多进程的话直接换成ProcessPoolExecutor即可
	with futures.ThreadPoolExecutor(workers) as executor:
		#map函数返回各个函数的返回值
		res = executor.map(download_one, cc_list)

	return len(list(res))



'''
期物(future)
封装期待完成的操作，可以放入队列
完成的状态可以查询，得到结果后可以获取结果
done方法不阻塞，返回值是布尔值
add_done_callback为期物添加回调函数
result在期物运行结束后调用将返回可调用对象的结果


标准库中的期物
concurrent.futures.Future
asyncio.Future
'''

def download_many(cc_list):

	with futures.ThreadPoolExecutor(max_workers=3) as Executor:
		to_do = []
		for cc in cc_list:
			#Executor.submit方法返回一个期物，表示待执行的操作
			future = Executor.submit(download_one,cc)
			todo.append(future)

		results = []
		#futures.as_completed方法传入期物期物列表，返回期物迭代器
		for future in futures.as_completed(to_do):
			res = future.result()
			msg = '{} result: {!r}'
			print(msg.format(future, res))
			results.append(res)

	return len(result)


'''
使用asyncio模块和aiohttp
'''
import asyncio
import collections

import aiohttp
from aiohttp import web
import tqdm

@asyncio.coroutine
def get_flag(cc):

	url = ''
	#异步操作请求url并让出
	resp = yield from aiohttp.request('GET',url)
	if resp.status == 200:
		image = yield from resp.read()
		return image
	else:
		raise web.HTTPNotFound()

@asyncio.coroutine
def download_one(cc, base_url, semaphore, verbose):

	try:
		#在yield from 表达式中把semaphore当成上下文管理器来使用，防止阻塞整个系统
		#退出这个with语句后semaphore计数器的值会递减，解除阻塞可能在等待同一个semaphore对象的
		#其他协程实例，semaphore对象维护着一个内部计数器，若在对象上调用.acquire()协程方法
		#计数器则递减，若在对象上调用.release()协程方法，计数器则递增
		#若计数器等于零时调用acquire不会阻塞，而是运行其他协程release让计数器递增
		with (yield from semaphore):
			image = yield from get_flag(cc)

	except web.HTTPNotFound:
		#dosomething
	except Exception as exc:
		#dosomething
	else:
		#save_flag应该要异步操作的，但是asyncio模块没有关于文件的操作
		#若有需要可以使用loop.run_in_executor在线程池中运行save_flag
		#asyncio的事件循环在背后维护着一个ThreadPoolExecutor对象
		#我们可以调用run_in_executor方法把可调用对象发给它执行
		#第一个参数是Executor实例，如果设为None则默认ThreadPoolExecutor
		loop = asyncio.get_event_loop()
		loop.run_in_executor(None,
				save_flag, image, cc.lower()+'.gif')

	if verbose:
		print()

	return cc

@asyncio.coroutine
def download_coro(cc_list, base_url, verbose, concur_req):

	counter = collections.Counter()
	semaphore = asyncio.Semaphore(concur_req)
	to_do = [download_one(cc, base_url, semaphore, verbose)
				for cc in sorted(cc_list)]

	#返回一个已完成期物的迭代器
	to_do_iter = asyncio.as_completed(to_do)
	if not verbose:
		to_do_iter = tqdm.tqdm(to_do_iter, total=len(cc_list))

	for future in to_do_iter:
		try:
			res = yield from future
			status = 'a'#
		except Exception:
			#dosomething
		else:
			counter[status] += 1

	return counter

def download_main(cc_list, base_url, verbose, concur_req):

	loop = asyncio.get_event_loop()
	coro = download_coro(cc_list, base_url, verbose, concur_req)
	counts = loop.run_until_complete(coro)
	loop.close()

	return counts


'''
tcp协程
'''
index = UnicodeNameIndex()
#参数分别是StreamReader, StreamWriter的实例
#写出数据时要字节类型
@asyncio.coroutine
def handle_queries(reader,writer):

	while True:
		#由于写入缓存所以不用异步
		writer.write(b'?>')
		#从缓存写入到用户
		yield from writer.drain()
		data = yield from reader.readline()
		try:
			query = data.decode().strip()
		except UnicodeDecodeError:
			query = '\x00'
		#获取接入这个服务器的套接字信息
		client = writer.get_extra_info('peername')
		print('Received from {}: {!r}'.format(client, query))

		if query:
			if ord(query[:1]) < 32:
				break

			lines = list(index.find_description_strs(query))
			if lines:
				writer.writelines(line.encode() + b'\r\n' for line in lines)
			writer.write(index.status(query, len(lines)).encode() + b'\r\n')

			yield from writer.drain()
			print('Send {} results'.format(len(lines)))
	print('Close the client socket')
	writer.close()

def main(address='127.0.0.1', port=2323):

	loop = asyncio.get_event_loop()
	#start_server返回协程，需要用yield from或者run_until_complete这样的函数来驱动
	coro = asyncio.start_server(loop)
	server = asyncio.run_until_complete(coro)

	try:
		loop.run_forever()
	except KeyboardInterrupty:
		pass

	server.close()
	loop.run_until_complete(server.wait_closed())
	loop.close()


'''
http服务器 协程
'''

def home(request):

	query = request.GET.get('query', ' ').strip()
	print('Query {!r}'.format(query))
	if query:
		descriptions = list(index.find_description_strs(query))
		#省略
		res = '\n'.join()
		msg = ''

	else:
		descriptions = []
		res = ''
		msg = ''

	html = template.format(query=query, result=res,
							message=msg)
	print('Sending {} results'.format(len(descriptions)))

	return web.Response(content_type=CONTENT_TYPE, text=html)

def init(loop, address, port):

	app = web.Application(loop=loop)
	app.route.add_route('GET', '/', home)
	handler = app.make_handler()
	server = yield from loop.create_server(handler, address, port)
	return server.sockets[0].getsockname()

def main(address='127.0.0.1', port=8888):

	port = int(port)
	loop = asyncio.get_event_loop()
	#init是协程，需要yield from或者类似run_until_complete这样的函数来驱动
	host = asyncio.run_until_complete(init(loop, address, port))

	try:
		loop.run_forever()
	except KeyboardInterrupt:
		pass

	loop.close()

