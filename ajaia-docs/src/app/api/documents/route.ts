import { NextRequest, NextResponse } from 'next/server';
import { Storage } from '@/lib/storage';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId') || undefined;
    const filter = (searchParams.get('filter') as 'all' | 'owned' | 'shared') || 'all';

    const documents = Storage.list(userId, filter);
    return NextResponse.json({ success: true, documents });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message || 'Failed to list documents' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { title, content, ownerId } = body;

    if (!ownerId) {
      return NextResponse.json({ success: false, error: 'ownerId is required to create a document' }, { status: 400 });
    }

    const newDoc = Storage.create({ title, content, ownerId });
    return NextResponse.json({ success: true, document: newDoc }, { status: 201 });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message || 'Failed to create document' }, { status: 500 });
  }
}
