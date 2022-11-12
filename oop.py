'''
Created on 15-10-2012

@author: Krzysiek
'''

import datetime

class Employee:

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@mail.com'
        self.pay = pay

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last
    
    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None

    def __add__(self, other):
        return self.pay + other.pay

    def __len__(self):
        return len(self.fullname())    

    def __str__(self):
        return '{} {} {}'.format(self.first, self.last, self.email)

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Steve-Smith-30000'
emp_str_3 = 'Jane-Doe-90000'       

emp_1 = Employee.from_string(emp_str_1)
print(emp_1)
emp_1.fullname = "Corey Schafer"
print(emp_1.fullname)

emp_2 = Employee.from_string(emp_str_2)
print(emp_2)

emp_3 = Employee.from_string(emp_str_3)
print(emp_3)

my_date = datetime.date(2016, 7, 11)
print(Employee.is_workday(my_date))

print(emp_1.__dict__)
print(emp_2.__dict__)