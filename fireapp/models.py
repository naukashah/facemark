from django.db import models


# Create your models here.
class UserDetails(models.Model):
    User_name = models.CharField(max_length=300)
    User_phone = models.BigIntegerField()
    User_address = models.TextField()
    User_pic = models.FileField(upload_to='documents/%Y/%m/%d')


class loginDetails(models.Model):
    name = models.CharField(max_length=300, blank=False, null=False)
    password = models.CharField(max_length=300, blank=False, null=False)
