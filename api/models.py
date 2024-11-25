from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10)
    vector_store_path = models.CharField(max_length=500, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class DocumentQuery(models.Model):
    question = models.TextField()
    answer = models.TextField()
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='queries')
    created_at = models.DateTimeField(auto_now_add=True)
    perplexity_answer = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.question
