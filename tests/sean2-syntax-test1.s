.global main
main:
	pushl %ebp
	movl %esp, %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	movl $3, %esi
	shll $2, %esi
	orl $0, %esi
	movl $0, %eax
	popl %esi
	popl %edi
	popl %ebx
	leave
	ret
