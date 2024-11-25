from django.shortcuts import render
from graphene_django.views import GraphQLView
from asgiref.sync import async_to_sync

# Create your views here.

class AsyncGraphQLView(GraphQLView):
    def execute_graphql_request(self, *args, **kwargs):
        result = super().execute_graphql_request(*args, **kwargs)
        if result is not None and hasattr(result, '__await__'):
            return async_to_sync(lambda: result)()
        return result
