from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.dispatch import receiver

from django.db.models.signals import post_save
# Create your models here.
class BaseModel(models.Model):
    is_deleted = models.BooleanField(default=False)
    created_time = models.DateTimeField(null=True)
    modified_time = models.DateTimeField(null=True)
    deleted_time = models.DateTimeField(null=True)

    class Meta:
        abstract = True

    # Override save method.
    def save(self,  *args, **kwargs):
        if not self.created_time:
            self.created_time = timezone.now()
        self.modified_time = timezone.now()

        super(BaseModel, self).save(*args, **kwargs)


class UserToken(BaseModel):
    user = models.OneToOneField(User, related_name="token", on_delete=models.DO_NOTHING)
    token = models.TextField(max_length=500, null=True, blank=True)

    def __str__(self):
        return "{}".format(self.user.email)

                                                                                                                                                                                                                                                                                                                                                                                                                                                       
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    profile_image = models.ImageField(default='default.png')
    

    def __str__(self):
        return self.user.username 

@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()


