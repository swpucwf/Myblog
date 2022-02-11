class Node(object):
	"""docstring for Node"""
	def __init__(self, val):
		super(Node, self).__init__()
		self.val = val
		self.next = None



class  LinkedList(object):
	"""LinkedList"""
	def __init__(self):
		super( LinkedList, self).__init__()
		self.head = Node(0)
		self.length = 0 


	def get(self,index):
		if index>=self.length or index<0:
			reutrn  -1

		count = 0
		temp_Node = self.head

		while count<index:
			count+=1
			temp_Node = temp_Node.next

		return temp_Node.next.val

	def __len__(self):
		'''
		return the length of LinkedList
		'''
		return self.length


	def AddAtidnex(self,index,val):
		
		if index>self.length or index<0:
			reutn -1

		count = 0
		temp_Node = self.head


		while count<index:
			count+=1
			temp_Node= temp_Node.next

		next_node = temp_Node.next
		node = Node(val)
		temp_Node.next = node
		node.next = next_node
		self.length+=1


	def deleteAtindex(self,idnex):
		
		if index<0 or index>=self.length:
			return -1

		count=0
		temp_Node = self.head

		while count<index:
			count+=1
			temp_Node = temp_Node.next

		next_node = temp_Node.next.next

		temp_Node.next = next_node

		self.length-=1






a

		