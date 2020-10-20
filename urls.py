from django.contrib import admin
from django.conf.urls import url, include
from django.conf import settings
from django.views.static import serve
from face import views
from django.conf.urls.static import static
urlpatterns = [
                  url(r'^$', views.home),
                  url(r'^admin/', admin.site.urls),
                  url(r'^clear', views.clear),
                  url(r'detect/',views.detect),
                  url(r'extract/', views.extract),
                  url(r'train/',views.train),
                  
                  # Url to select a file for the predictions
                  # url('fileselect/', SelectPredFileView.as_view(), name='file_select'),
                  #(? url(r'^delete/$', FileDeleteView.as_view(), name='APIdelete'),)
   
                
                  url(r'deleteuser/', views.deleteuser),
                  url(r'detectlive/', views.detectlive),
                  url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
