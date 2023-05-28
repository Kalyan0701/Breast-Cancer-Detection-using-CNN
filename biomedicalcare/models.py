from django.db import models


GENDER_CHOICES = (
    ('Male', 'Male'),
    ('Female', 'Female'),
)


class Patient(models.Model):
    # user_id = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=50)
    lastname = models.CharField(max_length=50)
    age = models.IntegerField()
    address = models.CharField(max_length=100)
    contactnumber = models.CharField(max_length=13)
    date = models.DateField()
    filename = models.ImageField(upload_to='images')
    imgclassify = models.CharField(max_length=50, null=True, blank=True)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)


class Describe(models.Model):
    Benign = models.CharField(max_length=500)
    Malignant = models.CharField(max_length=500)
    Normal = models.CharField(max_length=500)
