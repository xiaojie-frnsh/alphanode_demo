# Generated by Django 5.1.3 on 2024-11-25 03:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_document_query_delete_post'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='file',
            field=models.FileField(upload_to='uploads/'),
        ),
    ]