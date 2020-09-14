from django.db import models
import datetime

class userdata(models.Model):
    Video = models.FileField(upload_to='media/')
    User = models.CharField(max_length=100)
    Date_joined = models.DateField()

    def __str__(self):
        return self.User, self.Video, self.Date_joined
        